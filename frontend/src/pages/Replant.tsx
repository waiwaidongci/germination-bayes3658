import {useMemo,useState} from 'react';
import {useSearchParams,Link} from 'react-router-dom';
import {ArrowRight,Lock,Snowflake} from 'lucide-react';
import {useResearchData} from '../hooks/useResearch';
import {useAuth} from '../stores/auth';
import {useAction} from '../hooks/useAction';
import {createReplantPlan} from '../api/replant';
import {Notice} from '../components/common/Notice';
import {ReplantStatus,InfeasibleReason} from '../constants/enums';
import {number,percent,time} from '../utils/format';
import {ReplantPlan} from '../types/domain';

interface LotForm{lot_code:string;available_seeds:string;cost_per_seed:string}

export function Replant(){
  const s=useResearchData(),{user}=useAuth(),a=useAction(),[params]=useSearchParams();
  const runs=s.runs.filter(x=>x.status==='completed');
  const [runId,setRunId]=useState(Number(params.get('run'))||runs[0]?.id||0);
  const run=runs.find(x=>x.id===runId)||runs[0];
  const [name,setName]=useState('');
  const [target,setTarget]=useState(800);
  const [requiredProb,setRequiredProb]=useState(0.9);
  const [timeLimit,setTimeLimit]=useState(5);
  const [forms,setForms]=useState<Record<string,LotForm>>({});
  const [selectedPlanId,setSelectedPlanId]=useState(0);

  const lotForms=useMemo<LotForm[]>(()=>{
    if(!run?.result?.lots)return [];
    return run.result.lots.map(l=>{
      const existing=forms[l.lot.code];
      return existing??{lot_code:l.lot.code,available_seeds:'100',cost_per_seed:'1.00'};
    });
  },[run,forms]);

  const plans=s.plans.filter(p=>p.run_id===run?.id);
  const selectedPlan=s.plans.find(p=>p.id===selectedPlanId)||plans[0];

  function update(code:string,field:keyof Omit<LotForm,'lot_code'>,value:string){
    setForms(f=>({...f,[code]:{lot_code:code,available_seeds:f[code]?.available_seeds??'100',cost_per_seed:f[code]?.cost_per_seed??'1.00',[field]:value}}));
  }

  const validationError=useMemo(()=>{
    if(!run)return '没有已完成研究，无法制定补种方案';
    if(!Number.isInteger(target)||target<0)return '总萌发目标须为非负整数';
    if(!(requiredProb>0&&requiredProb<1))return '达标概率须在(0,1)之间';
    if(!(timeLimit>0&&timeLimit<=60))return '搜索时限须在(0,60]秒';
    for(const f of lotForms){
      const cap=Number(f.available_seeds),cost=Number(f.cost_per_seed);
      if(!Number.isInteger(cap)||cap<0||cap>200)return `批次 ${f.lot_code} 可用粒数须为0–200整数`;
      if(!(cost>0)||!Number.isFinite(cost))return `批次 ${f.lot_code} 每粒成本须为正数`;
    }
    return '';
  },[run,target,requiredProb,timeLimit,lotForms]);

  async function submit(){
    if(!run||validationError)return;
    const created=await createReplantPlan({
      name:name||`补种方案 · ${run.name}`,run_id:run.id,target,
      required_probability:requiredProb,time_limit:timeLimit,
      lots:lotForms.map(f=>({lot_code:f.lot_code,available_seeds:Number(f.available_seeds),cost_per_seed:Number(f.cost_per_seed)})),
    });
    setName('');setSelectedPlanId(created.id);await s.refresh();
  }

  return <div className="page replant-page">
    <div className="page-title">
      <div>
        <p className="eyebrow">REPLANT OPTIMIZATION / 06</p>
        <h1>补种最低成本整数分配<span>沿用冻结后验与批间独立；仅界限闭合才标最优。</span></h1>
      </div>
      {run&&<Link className="outline" to={`/notes?run=${run.id}`}>文字补种建议 →</Link>}
    </div>
    <Notice message={s.error||a.error} onRetry={s.refresh}/>

    <label>关联已完成研究
      <select aria-label="关联已完成研究" value={run?.id??''} onChange={e=>{setRunId(Number(e.target.value));setForms({})}}>
        {runs.map(x=><option key={x.id} value={x.id}>#{x.id} / {x.name}</option>)}
      </select>
    </label>
    {!run?<p className="empty">请先在先验与情景页完成一次批次概率研究。</p>:
    <>
      <div className="replant-grid">
        <section>
          <div className="section-heading"><h2>逐批可用粒数与每粒成本</h2><span>{lotForms.length}批 · 每批0–200粒</span></div>
          <div className="compact-table">
            <table>
              <thead><tr><th>批次</th><th>冻结后验 a / b</th><th>可用粒数</th><th>每粒成本</th><th>后验萌发率</th></tr></thead>
              <tbody>
                {lotForms.map((f,i)=>{
                  const lot=run.result.lots[i];
                  return <tr key={f.lot_code}>
                    <td>{f.lot_code}</td>
                    <td>{number(lot.post_alpha,2)} / {number(lot.post_beta,2)}</td>
                    <td><input aria-label={`${f.lot_code} 可用粒数`} type="number" min={0} max={200} step={1} value={f.available_seeds} disabled={user?.role==='viewer'} onChange={e=>update(f.lot_code,'available_seeds',e.target.value)}/></td>
                    <td><input aria-label={`${f.lot_code} 每粒成本`} type="number" min={0.0001} step={0.01} value={f.cost_per_seed} disabled={user?.role==='viewer'} onChange={e=>update(f.lot_code,'cost_per_seed',e.target.value)}/></td>
                    <td>{percent(lot.mean)}</td>
                  </tr>;
                })}
              </tbody>
            </table>
          </div>
        </section>

        <aside className="replant-controls">
          <h2>目标与搜索约定</h2>
          <label>方案名称<input aria-label="方案名称" value={name} maxLength={160} placeholder={`补种方案 · ${run.name}`} onChange={e=>setName(e.target.value)}/></label>
          <label>合计萌发目标 T（粒）<input aria-label="合计萌发目标" type="number" min={0} step={1} value={target} onChange={e=>setTarget(Math.floor(Number(e.target.value)))}/></label>
          <label>达标概率要求 Pr(S≥T)<input aria-label="达标概率要求" type="number" min={0.001} max={0.999} step={0.01} value={requiredProb} onChange={e=>setRequiredProb(Number(e.target.value))}/></label>
          <label>搜索时限（秒，超时降级）<input aria-label="搜索时限" type="number" min={0.01} max={60} step={0.5} value={timeLimit} onChange={e=>setTimeLimit(Number(e.target.value))}/></label>
          <p className="caption">目标函数 min Σ成本×粒数，约束 Pr(合计萌发≥T)≥要求值。各批萌发为冻结 Beta-Binomial、批间独立；超时返回已证成本下界与可行方案上界。</p>
          {validationError&&<p className="role-note warning">{validationError}</p>}
          {user?.role!=='viewer'
            ?<button className="primary" disabled={a.busy||!!validationError} onClick={()=>a.execute(submit)}>求解最低成本分配 <ArrowRight size={15}/></button>
            :<p className="role-note">阅读者可核对方案与界限，创建需要分析员权限。</p>}
        </aside>
      </div>

      <div className="section-heading"><h2>已保存方案（冻结，不随来源变化）</h2><span>{plans.length}份 · 选定研究</span></div>
      <label>查看方案
        <select aria-label="查看方案" value={selectedPlan?.id??''} onChange={e=>setSelectedPlanId(Number(e.target.value))}>
          {plans.map(p=><option key={p.id} value={p.id}>#{p.id} / {p.name} / {ReplantStatus[p.status]}</option>)}
        </select>
      </label>
      {selectedPlan?<PlanDetail plan={selectedPlan}/>:<p className="empty">尚无保存方案。提交求解后结果与输入快照一并冻结保存。</p>}
    </>}
  </div>;
}

function PlanDetail({plan}:{plan:ReplantPlan}){
  const r=plan.result;
  return <div className="plan-detail">
    <div className="plan-status-row">
      <span className={`replant-badge ${plan.status}`}><i/>{ReplantStatus[plan.status]}</span>
      <span className="eyebrow">{r.method}</span>
      <span className="freeze-tag"><Snowflake size={12}/> 快照 {plan.frozen_snapshot.snapshot_hash.slice(0,12)}… / FIT {plan.frozen_snapshot.fit_hash.slice(0,12)}…</span>
    </div>

    {plan.status==='infeasible'
      ? <div className="boundary-warning"><b>已证无解：</b>{InfeasibleReason[r.infeasible_reason as keyof typeof InfeasibleReason]||r.infeasible_reason}
          {r.max_tail_probability!=null&&<>；全部满配实际尾概率仅 {number(r.max_tail_probability,8)}</>}
        </div>
      : <div className="plan-metrics">
          <Metric label="状态判据" value={r.optimal?'界限已闭合，标最优':'界限未闭合，不标最优'} raw/>
          <Metric label="总成本（上界）" value={number(r.upper_bound??undefined,4)}/>
          <Metric label="已证成本下界" value={r.lower_bound==null?'—':number(r.lower_bound,4)}/>
          <Metric label="界限间隙" value={r.gap==null?'—':number(r.gap,4)}/>
          <Metric label="实际达标概率" value={percent(r.actual_probability??undefined)}/>
          <Metric label="概率要求" value={percent(r.required_probability)}/>
          <Metric label="合计补种 / 可用" value={`${r.total_seeds} / ${r.total_capacity}`} raw/>
          <Metric label="期望合计萌发" value={number(r.total_expected_germination,2)}/>
          <Metric label="搜索节点 / 超时" value={`${r.elapsed_nodes} / ${r.timed_out?'是':'否'}`} raw/>
        </div>}

    {plan.status!=='infeasible'&&<>
      <p className="caption">
        {r.timed_out
          ?'达到时限或节点上限提前返回：下界来自已剪支与未决前沿，上界方案已验证满足概率要求；二者未闭合，故标记“搜索未决”而非最优。'
          :'分支定界完整闭合：成本下界与可行上界相等，方案证明为最低成本整数分配。'}
        <Lock size={11}/> 方案保存的是冻结后验与逐批输入，后续研究或来源数据变化不会改写本方案。
      </p>
      <div className="compact-table">
        <table>
          <thead><tr><th>批次</th><th>分配粒数 / 可用</th><th>每粒成本</th><th>该行成本</th><th>期望萌发</th><th>冻结 a / b</th></tr></thead>
          <tbody>
            {r.allocation.map(row=><tr key={row.lot_code} className={row.seeds>0?'selected':''}>
              <td>{row.lot_code}</td>
              <td>{row.seeds} / {row.cap}</td>
              <td>{number(row.cost_per_seed,3)}</td>
              <td>{number(row.line_cost,3)}</td>
              <td>{number(row.expected_germination,3)}</td>
              <td>{number(row.post_alpha,2)} / {number(row.post_beta,2)}</td>
            </tr>)}
          </tbody>
        </table>
      </div>
    </>}
    <p className="fingerprint">{plan.owner} · {time(plan.created_at)} · {r.assumptions}</p>
  </div>;
}

function Metric({label,value,raw}:{label:string;value:string;raw?:boolean}){
  return <div className="metric"><span>{label}</span><b className={raw?'raw':''}>{value}</b></div>;
}
