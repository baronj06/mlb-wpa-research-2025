"""Independent proof-of-concept WE and expected-outcome LI, not provider stats.

No source probabilities enter fitting. Models are deliberately small and
versioned separately from EOWS. Unsupported transitions remain missing.
"""

from collections import Counter, defaultdict
from dataclasses import replace
from datetime import datetime

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from .attribution import State, move_runners, RUNNER_TYPES
from .reference_review import opportunity_v04, ledger_v04

FEATURE_NAMES=['intercept','home_score_diff_horizon_scaled','home_batting','remaining_half_innings_scaled',
    *[f'inning_{i}' for i in range(2,10)],'automatic_runner_extra_regime',
    *[f'signed_base_state_{i}_horizon_scaled' for i in range(1,8)],
    'signed_one_out_horizon_scaled','signed_two_outs_horizon_scaled']


def feature_vector(s):
    if s.outs not in (0,1,2) or s.inning<1 or s.half not in ('top','bottom'):
        raise ValueError('WE requires an active pre-event state')
    bottom=int(s.half=='bottom');extra=int(s.inning>=10)
    horizon=(2 if extra else 2*(10-s.inning))-bottom
    scale=1/np.sqrt(horizon);sign=1 if bottom else -1
    mask=sum((1<<i) for i,b in enumerate(s.bases) if b is not None)
    return np.array([1,(s.home_score-s.away_score)*scale,bottom,horizon/18,
                     *[int(s.inning==i) for i in range(2,10)],extra,
                     *[sign*scale*int(mask==b) for b in range(1,8)],
                     *[sign*scale*int(s.outs==o) for o in (1,2)]],dtype=float)


def normalize_state(s):
    """Deterministic game rules, not an observed winner label or fake PA."""
    if s.outs not in (0,1,2,3):raise ValueError('Invalid outs')
    if s.half=='bottom' and s.inning>=9 and s.home_score>s.away_score:
        return None,1.0
    if s.outs<3:return s,None
    if s.half=='top' and s.inning>=9 and s.home_score>s.away_score:return None,1.0
    if s.half=='bottom' and s.inning>=9 and s.home_score!=s.away_score:
        return None,float(s.home_score>s.away_score)
    inning=s.inning+(s.half=='bottom');half='bottom' if s.half=='top' else 'top'
    # Dummy occupancy tokens here are model state symbols, never player IDs.
    bases=(None,-2,None) if inning>=10 else (None,None,None)
    return replace(s,inning=inning,half=half,outs=0,bases=bases),None


def game_balanced_weights(games):
    games=np.asarray(games);counts=Counter(games)
    return np.array([len(games)/(len(counts)*counts[g]) for g in games])


class RidgeWE:
    def __init__(self,penalty=10.):
        self.penalty=penalty;self.calibration=np.array([0.,1.])

    def fit(self,states,winners,games):
        X=np.stack([feature_vector(s) for s in states]);y=np.asarray(winners,float)
        w=game_balanced_weights(games);self.training_games=set(games)
        def objective(beta):
            z=X@beta;p=expit(z);reg=beta.copy();reg[0]=0
            loss=np.sum(w*(np.logaddexp(0,z)-y*z))+.5*self.penalty*(reg@reg)
            grad=X.T@(w*(p-y))+self.penalty*reg
            return loss,grad
        bounds=[(None,None)]*X.shape[1];bounds[1]=(0,None)
        result=minimize(objective,np.zeros(X.shape[1]),jac=True,method='L-BFGS-B',bounds=bounds,
                        options={'maxiter':1000,'ftol':1e-11})
        if not result.success:raise RuntimeError('WE optimizer failed: '+result.message)
        self.beta=result.x;self.optimization={'success':True,'iterations':int(result.nit)}
        self.baseline=float(np.mean([y[np.asarray(games)==g][0] for g in set(games)]))
        return self

    def calibrate(self,states,winners,games):
        if self.training_games & set(games):raise ValueError('Calibration games overlap training')
        z=np.stack([feature_vector(s) for s in states])@self.beta;y=np.asarray(winners,float)
        w=game_balanced_weights(games);X=np.column_stack([np.ones(len(z)),z])
        def objective(b):
            zz=X@b;prior=b-np.array([0.,1.])
            return (np.sum(w*(np.logaddexp(0,zz)-y*zz))+.5*(prior@prior),
                    X.T@(w*(expit(zz)-y))+prior)
        r=minimize(objective,[0.,1.],jac=True,bounds=[(None,None),(0,None)],method='L-BFGS-B')
        if not r.success:raise RuntimeError('Calibration optimizer failed')
        self.calibration=r.x;self.calibration_games=set(games)
        return self

    def predict(self,states,calibrated=True):
        out=[]
        for s in states:
            active,terminal=normalize_state(s)
            if terminal is not None:out.append(terminal);continue
            z=float(feature_vector(active)@self.beta)
            if calibrated:z=self.calibration[0]+self.calibration[1]*z
            out.append(float(expit(z)))
        return np.asarray(out)


def collapse_moves(runners):
    groups=defaultdict(list)
    for r in runners:groups[r['details']['runner']['id']].append(r)
    result=[]
    for moves in groups.values():
        for a,b in zip(moves,moves[1:]):
            if a['movement'].get('end')!=b['movement'].get('start'):
                raise ValueError('ambiguous_terminal_movement_chain')
        last=moves[-1]
        result.append({**last,'movement':{**last['movement'],'start':moves[0]['movement'].get('start')}})
    return result


def extract_game(game_pk,meta,plays,role):
    """Use official identities/moves; do not read supplied numeric WPA/LI/WE."""
    board=meta['liveData']['linescore'];home=board['teams']['home']['runs'];away=board['teams']['away']['runs']
    if home==away:raise ValueError('Nonfinal/tied game')
    shortened=board['currentInning']<9
    rows=[]
    for pos,p in enumerate(plays):
        credits={c['player']['id'] for c in p.get('credits',[]) if c.get('credit')=='b_pa'}
        if not credits:continue
        row={'game_pk':game_pk,'at_bat_index':p['about']['atBatIndex'],'role':role,
             'batter_id':next(iter(credits)) if len(credits)==1 else None,
             'home_won':int(home>away),'result':p['result']['eventType'],
             'start_state':None,'batter_before':None,'after_state':None,
             'supported':False,'reason':'','runner_states':[]}
        try:
            if shortened:raise ValueError('unsupported_shortened_game')
            state,_=opportunity_v04(plays,pos)
            verify_start_boundary(p)
            row['start_state']=state
            units,blockers=ledger_v04([p],{p['about']['atBatIndex']:state})
            if blockers:raise ValueError(';'.join(sorted({b['reason'] for b in blockers})))
            batter=[u for u in units if u.kind=='batter']
            if len(batter)!=1:raise ValueError('ambiguous_batter_event')
            b=batter[0]
            terminal=[r for r in p.get('runners',[]) if r['details']['playIndex']==b.action_index]
            if not terminal:raise ValueError('missing_terminal_movements')
            after=move_runners(b.before,collapse_moves(terminal))
            official=(p['count']['outs'],p['result']['homeScore'],p['result']['awayScore'])
            if (after.outs,after.home_score,after.away_score)!=official:
                raise ValueError('terminal_post_state_totals_mismatch')
            expected=tuple(p['matchup'].get(k,{}).get('id') for k in ('postOnFirst','postOnSecond','postOnThird'))
            if after.outs<3 and after.bases!=expected:raise ValueError('terminal_post_state_bases_mismatch')
            own_runs=(after.home_score-b.before.home_score) if state.half=='bottom' else (after.away_score-b.before.away_score)
            other_runs=(after.away_score-b.before.away_score) if state.half=='bottom' else (after.home_score-b.before.home_score)
            if own_runs<0 or other_runs!=0:raise ValueError('invalid_score_transition')
            row.update(supported=True,batter_before=b.before,after_state=after,
                runner_states=[(u.before,units[i+1].before) for i,u in enumerate(units) if u.kind=='runner'],
                outcome=(own_runs,after.outs-b.before.outs,after.mask))
        except (ValueError,KeyError,TypeError) as exc:row['reason']=str(exc)
        rows.append(row)
    expected_pa=sum(p.get('stats',{}).get('batting',{}).get('plateAppearances',0)
                    for side in ('home','away') for p in meta['liveData']['boxscore']['teams'][side]['players'].values())
    if len(rows)!=expected_pa:raise ValueError('POC official denominator mismatch')
    return rows


def verify_start_boundary(play):
    """Do not label between-batter actions as belonging to a started PA."""
    for event in play['playEvents']:
        if event.get('details',{}).get('eventType') not in RUNNER_TYPES:continue
        prior=any(e.get('isPitch') and e['index']<event['index'] for e in play['playEvents'])
        explicit=event.get('details',{}).get('description','').startswith(
            f"With {play['matchup']['batter'].get('fullName','__unknown__')} batting,")
        try:
            start=play['about']['startTime'];when=event['startTime']
            after=datetime.fromisoformat(when.replace('Z','+00:00'))>=datetime.fromisoformat(start.replace('Z','+00:00'))
        except (KeyError,ValueError):after=False
        if not (prior or explicit) or not after:
            raise ValueError('runner_action_pa_boundary_unverified')


def counterfactual_state(start,outcome):
    runs,added_outs,mask=outcome
    if runs<0 or added_outs<0 or start.outs+added_outs>3:raise ValueError('Invalid counterfactual outcome')
    bases=tuple(-(i+1) if ch!='_' else None for i,ch in enumerate(mask))
    return replace(start,outs=start.outs+added_outs,bases=bases,
                   home_score=start.home_score+(runs if start.half=='bottom' else 0),
                   away_score=start.away_score+(runs if start.half=='top' else 0))


class ExpectedOutcomeLI:
    """Plug-in batting-outcome expectation, not abs(realized WPA)."""
    def __init__(self,model,min_pa=30,min_games=10):
        self.model=model;self.min_pa=min_pa;self.min_games=min_games;self.cache={}

    def fit(self,rows):
        self.counts=defaultdict(Counter);self.games=defaultdict(set)
        self.excluded_walkoff_donors=0
        for r in rows:
            if not r['supported']:continue
            if r['game_pk'] not in self.model.training_games:raise ValueError('Transition-kernel leakage')
            end=r['after_state']
            if end.half=='bottom' and end.inning>=9 and end.home_score>end.away_score:
                self.excluded_walkoff_donors+=1
                continue
            s=r['batter_before'];key=(s.outs,s.mask)
            self.counts[key][r['outcome']]+=1;self.games[key].add(r['game_pk'])
        raw=[self.expected_change(r['start_state'])[0] for r in rows if r['start_state'] is not None]
        raw=[v for v in raw if v is not None]
        if not raw or np.mean(raw)<=0:raise ValueError('No supported LI normalizer')
        self.normalizer=float(np.mean(raw));self.normalization_games=set(r['game_pk'] for r in rows)
        return self

    def expected_change(self,s):
        key=(s.outs,s.mask);counts=self.counts.get(key,{});n=sum(counts.values());ng=len(self.games.get(key,set()))
        info={'kernel_pa':n,'kernel_games':ng,'outcome_support':len(counts)}
        if n<self.min_pa or ng<self.min_games or len(counts)<2:return None,info
        bucket=(s.inning,s.half,s.outs,s.mask,s.home_score-s.away_score)
        if bucket not in self.cache:
            outcomes=list(counts);future=self.model.predict([counterfactual_state(s,o) for o in outcomes])
            before=self.model.predict([s])[0];probs=np.array([counts[o]/n for o in outcomes])
            self.cache[bucket]=(float(probs@np.abs(future-before)),float(probs@(future-before)))
        raw,drift=self.cache[bucket];info['expected_home_we_drift']=drift
        return raw,info

    def predict(self,s):
        raw,info=self.expected_change(s)
        return (None if raw is None else raw/self.normalizer),info


def attributed_wpa(model,before,after):
    if before.inning!=after.inning or before.half!=after.half:
        raise ValueError('Raw event states must precede boundary normalization')
    sign=1 if before.half=='bottom' else -1
    return float(sign*(model.predict([after])[0]-model.predict([before])[0]))


def structural_diagnostics(model):
    """Fixed baseball-state grid, no fitting or holdout outcome access."""
    outs_violations=[];base_violations=[];score_violations=[]
    for inning in range(1,11):
        for half in ('top','bottom'):
            sign=1 if half=='bottom' else -1
            for diff in range(-5,6):
                for mask in range(8):
                    bases=tuple(-(i+1) if mask&(1<<i) else None for i in range(3))
                    for outs in range(3):
                        s=State(inning,half,outs,bases,max(diff,0),max(-diff,0))
                        # Exclude already game-over bottom states.
                        if normalize_state(s)[1] is not None:continue
                        p=model.predict([s])[0]
                        score_violations.append(max(0,p-model.predict([replace(s,home_score=s.home_score+1)])[0]))
                        if outs<2:
                            outs_violations.append(max(0,sign*(model.predict([replace(s,outs=outs+1)])[0]-p)))
                        for b in range(3):
                            if mask&(1<<b):continue
                            added=list(bases);added[b]=-(b+1)
                            base_violations.append(max(0,sign*(p-model.predict([replace(s,bases=tuple(added))])[0])))
    result={}
    for name,values in [('score',score_violations),('more_outs',outs_violations),('added_baserunner',base_violations)]:
        result[name]={'comparisons':len(values),'violations_gt_1e_10':int(np.sum(np.asarray(values)>1e-10)),
                      'max_probability_violation':float(max(values,default=0))}
    return result
