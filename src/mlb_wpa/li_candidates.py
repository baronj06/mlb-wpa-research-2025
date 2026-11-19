"""Frozen structural WE/LI candidates; no reference values enter estimation."""

from collections import Counter, defaultdict

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit

from .independent_we import (RidgeWE, normalize_state, game_balanced_weights,
                             counterfactual_state)


MASKS=[''.join(str(i+1) if b&(1<<i) else '_' for i in range(3)) for b in range(8)]
KEYS=[(o,b) for o in range(3) for b in MASKS]


def canonical_outcome(key,outcome):
    runs,added,mask=outcome
    return (int(runs),int(added),'___' if key[0]+added==3 else mask)


def load_baseline(model_data):
    model=RidgeWE();model.beta=np.array(model_data['coefficient_vector'])
    model.calibration=np.array(model_data['calibration'])
    model.training_games=set(model_data['fit_game_ids'])
    model.calibration_games=set(model_data['calibration_game_ids'])
    counts={}
    for row in model_data['kernel']:
        key=(row['outs'],row['bases']);counter=Counter()
        for x in row['outcomes']:
            counter[canonical_outcome(key,(x['runs'],x['added_outs'],x['after_bases']))]+=x['count']
        counts[key]=counter
    return model,counts


class MarkovWE:
    """Run-count absorbing chain, finite regulation + stationary extra innings.

    The same empirical average-batter kernel is used for both teams. Home
    advantage is handled by separately fitted calibration, not test outcomes.
    """
    def __init__(self,counts,training_games,max_runs=40,tail_tolerance=1e-8):
        self.training_games=set(training_games);self.counts=counts
        self.calibration=np.array([0.,1.]);self.dist_cache={}
        lookup={s:i for i,s in enumerate(KEYS)};n=len(KEYS)
        matrices=defaultdict(lambda:np.zeros((n,n)));absorbed=np.zeros((max_runs+1,n))
        for key in KEYS:
            if key not in counts or sum(counts[key].values())<30:raise ValueError('Unsupported Markov parent state')
            total=sum(counts[key].values());i=lookup[key]
            for (runs,outs,bases),count in counts[key].items():
                probability=count/total;new_outs=key[0]+outs
                if runs<0 or runs>max_runs or new_outs>3:raise ValueError('Invalid run transition')
                if new_outs==3:absorbed[runs,i]+=probability
                else:matrices[runs][i,lookup[(new_outs,bases)]]+=probability
        A=np.eye(n)-matrices[0];pmf=np.zeros((n,max_runs+1))
        for r in range(max_runs+1):
            rhs=absorbed[r].copy()
            for scored,matrix in matrices.items():
                if 0<scored<=r:rhs+=matrix@pmf[:,r-scored]
            pmf[:,r]=np.linalg.solve(A,rhs)
        if pmf.min() < -1e-12:raise ValueError('Negative run probability')
        pmf=np.maximum(pmf,0);mass=pmf.sum(axis=1)
        self.max_truncated_tail=float(np.max(1-mass))
        if np.max(np.abs(1-mass))>tail_tolerance:raise ValueError('Run-distribution truncation too large')
        # Documented numerical renormalization bounded by the measured tail.
        pmf/=mass[:,None]
        self.run_pmf={key:pmf[i] for key,i in lookup.items()}
        self.full=[np.array([1.])]
        for _ in range(9):self.full.append(np.convolve(self.full[-1],self.run_pmf[(0,'___')]))

    def raw_probability(self,s):
        active,terminal=normalize_state(s)
        if terminal is not None:return terminal
        s=active;inning=min(s.inning,10);key=(inning,s.half,s.outs,s.mask)
        if key not in self.dist_cache:
            current=self.run_pmf[(s.outs,s.mask)]
            if inning==10:
                H=self.run_pmf[(0,'_2_')] if s.half=='top' else current
                A=current if s.half=='top' else np.array([1.])
            elif s.half=='top':
                H=self.full[10-inning];A=np.convolve(current,self.full[9-inning])
            else:
                H=np.convolve(current,self.full[9-inning]);A=self.full[9-inning]
            difference=np.convolve(H,A[::-1]);difference/=difference.sum()
            self.dist_cache[key]=(np.cumsum(difference),difference,len(A)-1)
        cdf,probability,offset=self.dist_cache[key]
        threshold=-(s.home_score-s.away_score)+offset
        if threshold<0:return 1.
        if threshold>=len(cdf):return 0.
        # A tied completed regulation/extra inning returns to a symmetric
        # automatic-runner extra inning. Its uncalibrated home WE is 0.5.
        return float(np.clip(1-cdf[threshold]+.5*probability[threshold],0,1))

    def predict(self,states,calibrated=True):
        result=[]
        for s in states:
            _,terminal=normalize_state(s)
            p=self.raw_probability(s)
            if calibrated and terminal is None:
                p=float(expit(self.calibration[0]+self.calibration[1]*logit(np.clip(p,1e-12,1-1e-12))))
            result.append(p)
        return np.asarray(result)

    def calibrate(self,rows):
        ids={r['game_pk'] for r in rows}
        if ids&self.training_games:raise ValueError('Calibration leakage')
        z=logit(np.clip(self.predict([r['start_state'] for r in rows],False),1e-12,1-1e-12))
        X=np.column_stack([np.ones(len(z)),z]);y=np.array([r['home_won'] for r in rows])
        w=game_balanced_weights([r['game_pk'] for r in rows])
        def objective(b):
            v=X@b;delta=b-np.array([0.,1.])
            return (np.sum(w*(np.logaddexp(0,v)-y*v))+.5*(delta@delta),X.T@(w*(expit(v)-y))+delta)
        result=minimize(objective,[0.,1.],jac=True,method='L-BFGS-B',bounds=[(None,None),(0,None)])
        if not result.success:raise ValueError('Markov calibration failed')
        self.calibration=result.x;self.calibration_games=ids
        return self


def context_key(s):
    phase='early' if s.inning<=6 else 'late' if s.inning<=9 else 'extra'
    own_diff=(s.home_score-s.away_score)*(1 if s.half=='bottom' else -1)
    return (s.outs,s.mask,phase,s.half,int(np.sign(own_diff)))


class PotentialLI:
    """Expected absolute changes under fixed global or hierarchical outcomes."""
    def __init__(self,model,parents,training_rows,hierarchical=False,prior_pa=200):
        self.model=model;self.parents=parents;self.child=defaultdict(Counter)
        self.hierarchical=hierarchical;self.prior_pa=prior_pa;self.cache={}
        counted=defaultdict(Counter)
        for row in training_rows:
            if row['game_pk'] not in model.training_games:raise ValueError('LI training leakage')
            if not row['supported']:continue
            end=row['after_state']
            if end.half=='bottom' and end.inning>=9 and end.home_score>end.away_score:continue
            s=row['batter_before'];key=(s.outs,s.mask);outcome=canonical_outcome(key,row['outcome'])
            counted[key][outcome]+=1;self.child[context_key(s)][outcome]+=1
        if dict(counted)!=parents:raise ValueError('Training outcome counts differ from frozen baseline')
        values=[self.expected(r['start_state'])[0] for r in training_rows if r['start_state'] is not None]
        self.normalizer=float(np.mean(values));self.normalization_games={r['game_pk'] for r in training_rows}
        if self.normalizer<=0:raise ValueError('Invalid LI normalizer')

    def probabilities(self,s):
        parent=self.parents[(s.outs,s.mask)];total=sum(parent.values());outcomes=list(parent)
        p=np.array([parent[o]/total for o in outcomes]);child=self.child[context_key(s)];n=sum(child.values())
        if self.hierarchical:
            p=(np.array([child[o] for o in outcomes])+self.prior_pa*p)/(n+self.prior_pa)
        if not np.isclose(p.sum(),1,atol=1e-12) or np.any(p<0):raise ValueError('Outcome probabilities not normalized')
        return outcomes,p,{'parent_pa':total,'context_pa':n,'hierarchical_weight':n/(n+self.prior_pa) if self.hierarchical else 0.}

    def expected(self,s):
        key=(min(s.inning,10),s.half,s.outs,s.mask,s.home_score-s.away_score)
        if key not in self.cache:
            outcomes,p,info=self.probabilities(s)
            after=self.model.predict([counterfactual_state(s,o) for o in outcomes]);before=self.model.predict([s])[0]
            self.cache[key]=(float(p@np.abs(after-before)),float(p@(after-before)),info)
        return self.cache[key]

    def predict(self,s):
        expected,drift,info=self.expected(s)
        return expected/self.normalizer,drift,info
