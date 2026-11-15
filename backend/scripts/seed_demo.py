import json
import numpy as np
from src.config.database import Base,engine,SessionLocal
from src.models import User,GerminationStudy,SeedLot,LotCount,BayesRun,SamplingNote,ReplantPlan
from src.utils.security import hash_password
from src.services.study_service import create_snapshot
from src.services.analysis_service import analyze_germination

def seed():
    Base.metadata.create_all(engine)
    with SessionLocal() as db:
        for role,name in [('admin','陆岑 · 种质负责人'),('analyst','许禾 · 批次分析员'),('viewer','温桥 · 抽样阅读者')]:
            email=role+'@duocen.demo'
            if not db.query(User).filter_by(email=email).first():db.add(User(email=email,name=name,role=role,password_hash=hash_password('Duocen2026!')))
        studies=[]
        for si,(code,name) in enumerate([('DC-01','异质批次 · 萌发观察'),('DC-02','同观测率 · 上界诊断'),('DC-03','全零计数 · 拒绝边界')]):
            study=db.query(GerminationStudy).filter_by(code=code).first()
            if not study:study=GerminationStudy(code=code,name=name,description='28个独立合成批次，试种数20–180；DC-01潜在批率Beta(8,4)后Binomial抽样；DC-02是全部70/100的同观测率人工边界夹具；DC-03全部0。条件EB不作为种子合格或发货保证。',source='duocen-germination-lots-202611.csv');db.add(study);db.flush()
            rng=np.random.default_rng(320+si);ns=rng.integers(20,181,size=28);ps=rng.beta(8,4,size=28);ks=rng.binomial(ns,ps)
            if si==1:ns[:]=100;ks[:]=70
            if si==2:ks[:]=0
            for i,(n,k) in enumerate(zip(ns,ks)):
                lc=f'{code}-L{i+1:02}';lot=db.query(SeedLot).filter_by(study_id=study.id,code=lc).first()
                if not lot:lot=SeedLot(study_id=study.id,code=lc,name=f'独立种子批次 {i+1:02}',origin='铎岑种质试验室 / 合成批次');db.add(lot);db.flush()
                if not db.query(LotCount).filter_by(lot_id=lot.id).first():db.add(LotCount(lot_id=lot.id,n=int(n),k=int(k),note='合成汇总；n为试种粒数，k为萌发粒数，未萌发n−k；每批一条计数。'))
            db.flush();studies.append(study)
        if not db.query(BayesRun).filter_by(name='异质批次 · 条件EB基线').first():
            study=studies[0];snapshot=create_snapshot(db,study);result=analyze_germination(snapshot,.7,50);run=BayesRun(name='异质批次 · 条件EB基线',study_id=study.id,threshold=.7,future_n=50,input_snapshot=snapshot,status='completed',result=result,owner='许禾 · 批次分析员');db.add(run);db.flush();db.add(SamplingNote(run_id=run.id,title='首批独立补种研究说明',interpretation='后验与同批未来计数条件于点估αβ，不含超参数估计误差；贴边必须标弱识别。',proposal='保留每批n/k口径，下一轮同批独立试种，不把阈值概率当成合格或自动发货指令。',status='approved',owner='许禾 · 批次分析员',reviewer='陆岑 · 种质负责人',review_note='批准研究归档；结果只供独立补种研究，条件EB不作种子合格保证。'))
        # 补种最低成本整数分配演示：固定逐批可用粒数/成本，方案与冻结快照一并保存
        baseline=db.query(BayesRun).filter_by(name='异质批次 · 条件EB基线').first()
        if baseline and not db.query(ReplantPlan).filter_by(run_id=baseline.id).first():
            from src.services.replant_service import solve_replant
            from src.services.replant_plan_service import PLAN_VERSION
            from src.utils.fingerprint import fingerprint as _fp
            lot_inputs=[{'code':row['lot']['code'],'cap':200,'cost':round(1.0+(i%7)*0.35,2),
                         'post_alpha':float(row['post_alpha']),'post_beta':float(row['post_beta'])}
                        for i,row in enumerate(baseline.result['lots'])]
            plan_result=solve_replant(lot_inputs,900,.9,3.0)
            frozen={'plan_version':PLAN_VERSION,'run_id':baseline.id,'run_name':baseline.name,
                    'fit_hash':baseline.result.get('fit_hash'),'input_hash':baseline.result.get('input_hash'),
                    'fit_alpha':baseline.result['fit']['alpha'],'fit_beta':baseline.result['fit']['beta'],
                    'threshold':baseline.threshold,'future_n':baseline.future_n,'study':baseline.result.get('study'),
                    'lots':[{'lot_code':x['code'],'available_seeds':x['cap'],'cost_per_seed':x['cost'],
                             'post_alpha':x['post_alpha'],'post_beta':x['post_beta'],
                             'lot':next(r['lot'] for r in baseline.result['lots'] if r['lot']['code']==x['code']),
                             'count':next(r['count'] for r in baseline.result['lots'] if r['lot']['code']==x['code'])}
                            for x in lot_inputs],
                    'target':900,'required_probability':.9,'time_limit':3.0}
            frozen['snapshot_hash']=_fp({'version':PLAN_VERSION,'run_input_hash':frozen['input_hash'],
                                         'fit_hash':frozen['fit_hash'],
                                         'lots':[{'code':x['lot_code'],'cap':x['available_seeds'],'cost':x['cost_per_seed'],
                                                  'a':x['post_alpha'],'b':x['post_beta']} for x in frozen['lots']],
                                         'target':900,'required_probability':.9})
            plan_result['snapshot_hash']=frozen['snapshot_hash']
            db.add(ReplantPlan(name='28批补种最低成本演示',run_id=baseline.id,target=900,
                               required_probability=.9,time_limit=3.0,status=plan_result['status'],
                               frozen_snapshot=frozen,result=plan_result,owner='许禾 · 批次分析员'))
        db.commit();print(json.dumps({m.__tablename__:db.query(m).count() for m in [User,GerminationStudy,SeedLot,LotCount,BayesRun,SamplingNote,ReplantPlan]},ensure_ascii=False))
if __name__=='__main__':seed()
