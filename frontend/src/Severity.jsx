import React, {useEffect, useMemo, useState} from 'react';
import {Bar, BarChart, CartesianGrid, Cell, Legend, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis} from 'recharts';
import {AlertTriangle, ArrowUpRight, CarFront, RefreshCw, RotateCcw, ShieldCheck, Sparkles} from 'lucide-react';

const todayISO = new Date().toISOString().slice(0, 10);
const pct = v => `${(Number(v || 0) * 100).toFixed(0)}%`;
const fx = (v, d = 3) => (v === null || v === undefined || Number.isNaN(Number(v)) ? '—' : Number(v).toFixed(d));
const short = (s, n = 52) => (s.length > n ? s.slice(0, n - 2) + '…' : s);
const EMPTY = {date: todayISO, time: '12:00', chainage_km: ''};

// Quick scenarios. Codes that do not exist in the loaded model are skipped automatically.
const PRESETS = [
  {label: 'Night, two-wheeler vs truck', values: {time: '23:30', area_type: '2', vehicle_1: '6', vehicle_2: '5', weather: '1'}},
  {label: 'Daytime, car vs car, clear', values: {time: '11:00', area_type: '2', vehicle_1: '1', vehicle_2: '1', weather: '1'}},
  {label: 'Pedestrian involved, fog', values: {time: '06:30', area_type: '2', vehicle_1: '9', vehicle_2: '1', weather: '2'}},
  {label: 'Cycle, heavy rain, evening', values: {time: '18:45', area_type: '2', vehicle_1: '8', vehicle_2: '5', weather: '5'}},
];

// factor name returned by the API -> key in the echoed input
const FACTOR_INPUT = {'Date (month, weekday)': 'Date', 'Time of day': 'Time', 'Location (chainage, road side)': 'Chainage (km)'};

async function getJSON(url, options) {
  const r = await fetch(url, options);
  const body = await r.json().catch(() => ({}));
  if (!r.ok) {
    const d = body.detail;
    throw new Error(typeof d === 'string' ? d : Array.isArray(d) ? d.map(x => x.msg).join('; ') : `Request failed (${r.status})`);
  }
  return body;
}

const ChartTip = ({active, payload, render}) => (active && payload && payload.length
  ? <div className="sev-tip">{render(payload[0].payload)}</div> : null);

export default function SeverityPage({api}) {
  const base = `${api}/v4/severity`;
  const [schema, setSchema] = useState(null);
  const [analysis, setAnalysis] = useState(null);
  const [loadError, setLoadError] = useState('');
  const [form, setForm] = useState(EMPTY);
  const [result, setResult] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    getJSON(`${base}/schema`).then(setSchema).catch(e => setLoadError(e.message));
    getJSON(`${base}/analysis`).then(setAnalysis).catch(() => setAnalysis(null));   // optional
  }, [base]);

  const set = (k, v) => setForm(f => ({...f, [k]: v}));
  const optionsOf = useMemo(() => Object.fromEntries((schema?.fields || []).map(f => [f.name, new Set(f.options.map(o => o.value))])), [schema]);

  const applyPreset = pr => {
    const next = {...EMPTY};
    Object.entries(pr.values).forEach(([k, v]) => {
      if (k === 'time' || k === 'date' || k === 'chainage_km') next[k] = v;
      else if (optionsOf[k]?.has(v)) next[k] = v;
    });
    setForm(next); setResult(null); setError('');
  };
  const reset = () => { setForm(EMPTY); setResult(null); setError(''); };

  const run = async () => {
    if (!form.date || !form.time) { setError('Enter both a date and a time.'); return; }
    setBusy(true); setError('');
    const body = {date: form.date, time: form.time};
    if (form.chainage_km !== '') body.chainage_km = Number(form.chainage_km);
    (schema?.fields || []).forEach(f => { if (form[f.name]) body[f.name] = form[f.name]; });
    try { setResult(await getJSON(`${base}/predict`, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)})); }
    catch (e) { setError(e.message); setResult(null); }
    setBusy(false);
  };
  const onKey = e => { if (e.key === 'Enter' && e.target.tagName !== 'SELECT' && !busy) run(); };

  if (loadError) return (
    <section className="sev-page"><div className="model-callout"><div className="insight-icon"><AlertTriangle/></div>
      <div><b>Severity model not available</b><p>{loadError}</p><p>From the project root run <code>python -m v4_pipeline.severity_v2</code>, then restart the API.</p></div></div></section>);
  if (!schema) return <section className="sev-page"><div className="table-empty">Loading severity model…</div></section>;

  const m = schema.model; const ci = m.test_roc_auc_95ci;
  const p = result?.prediction;
  const blank = schema.fields.filter(f => !form[f.name]).map(f => f.title.replace(/ \(.*\)/, ''));
  const contributions = result?.explanation?.contributions || [];
  const maxEffect = Math.max(0.01, ...contributions.map(c => Math.abs(c.effect)));

  // analysis data (only the model the ablation was actually run on)
  const ablModel = analysis?.ablation?.find(r => r.experiment === 'Full model')?.model;
  const ablRows = (analysis?.ablation || []).filter(r => r.model === ablModel);
  const ablationData = ablRows.filter(r => r.experiment.startsWith('Drop'))
    .map(r => ({name: r.experiment.replace('Drop: ', ''), delta: r.delta_vs_full, ci: r.delta_95ci, clear: r.delta_distinguishable_from_0}))
    .sort((a, b) => a.delta - b.delta);
  const tradeoff = (analysis?.threshold_tradeoff || []).map(r => ({share: r.flagged_share_of_all_crashes, severe: r.severe_recall, fatal: r.fatal_recall, random: r.flagged_share_of_all_crashes}))
    .sort((a, b) => a.share - b.share);
  const calib = (analysis?.calibration_deciles || []).map(r => ({pred: r.mean_predicted, obs: r.observed_severe_rate, rows: r.rows}));
  const cmpA = analysis?.model_comparison_binary || [];
  const cmpB = analysis?.model_comparison_fourclass || [];

  return (
    <section className="sev-page" onKeyDown={onKey}>
      <div className="intro">
        <div>
          <div className="eyebrow"><span/> SEVERITY OF A RECORDED CRASH</div>
          <h1>How severe is<br/><em>this crash likely to be?</em></h1>
          <p>Estimates whether a crash that has already happened is severe (fatal or grievous injury) from road, vehicle and time details. It does not predict whether or where crashes occur.</p>
        </div>
        <div className="intro-chip"><div className="chip-icon"><CarFront/></div>
          <div><small>MODEL{m.calibrated ? ' · CALIBRATED' : ''}</small><b>{short(m.name, 34)}</b></div></div>
      </div>

      <div className="model-callout"><div className="insight-icon"><ShieldCheck/></div>
        <div><b>Weak model: read results with care</b>
          <p>Test ROC-AUC {fx(m.test_roc_auc, 2)}{ci ? ` (95% CI ${fx(ci[0], 2)} to ${fx(ci[1], 2)})` : ''}, where 0.50 is a coin flip.
            {m.members ? ` Built from ${m.members.length} models averaged together.` : ''} Research prototype, not for emergency or operational decisions.</p></div></div>

      <div className="sev-presets"><span><Sparkles size={13}/> Try a scenario</span>
        {PRESETS.map(pr => <button key={pr.label} onClick={() => applyPreset(pr)}>{pr.label}</button>)}</div>

      <div className="sev-grid">
        <div className="table-card sev-form">
          <div className="card-head"><div><div className="eyebrow">CRASH DETAILS</div><h2>Describe the crash</h2></div>
            <button className="sev-link" onClick={reset}><RotateCcw size={12}/> Reset</button></div>
          <div className="sev-fields">
            <label><span>Date</span><input type="date" value={form.date} onChange={e => set('date', e.target.value)}/></label>
            <label><span>Time (24h)</span><input type="time" value={form.time} onChange={e => set('time', e.target.value)}/></label>
            <label><span>Chainage (km)</span><input type="number" min="0" step="0.1" placeholder={`${schema.chainage_range_km?.[0]?.toFixed(0)}–${schema.chainage_range_km?.[1]?.toFixed(0)} in data`} value={form.chainage_km} onChange={e => set('chainage_km', e.target.value)}/></label>
            {schema.fields.map(f => (
              <label key={f.name}><span>{f.title}</span>
                <select value={form[f.name] || ''} onChange={e => set(f.name, e.target.value)}>
                  <option value="">Not recorded (typical value)</option>
                  {f.options.map(o => <option key={o.value} value={o.value}>{o.label}</option>)}
                </select></label>))}
          </div>
          <div className="sev-actions"><button className="run-btn" onClick={run} disabled={busy}>{busy ? <RefreshCw className="spin" size={14}/> : <ArrowUpRight size={14}/>} {busy ? 'Scoring' : 'Estimate severity'}</button>
            <small>Press Enter to submit</small></div>
          {error && <div className="error sev-error">{error}</div>}
        </div>

        <div className="table-card sev-result">
          <div className="card-head"><div><div className="eyebrow">RESULT</div><h2>{p ? p.label : 'Awaiting input'}</h2></div></div>
          {!p && <div className="table-empty">Pick a scenario or fill in what is known, then press “Estimate severity”. Unknown fields are filled with the most typical value.</div>}
          {p && <div className="sev-body">
            {p.history_for_similar_scores && <div className="sev-headline"><strong>{pct(p.history_for_similar_scores.observed_severe_rate_for_similar_scores)}</strong>
              <span>of past test crashes with scores like this were actually severe<small>based on {p.history_for_similar_scores.from_test_crashes} similar crashes</small></span></div>}
            <div className={'risk-score ' + (p.flagged_severe ? '' : 'low')}>
              <div><small>MODEL SCORE · THRESHOLD {fx(p.threshold, 2)}</small><b>{p.flagged_severe ? 'At or above threshold' : 'Below threshold'}</b></div>
              <strong>{fx(p.score, 2)}</strong>
            </div>
            <div className="sev-meter" title="Score vs the decision threshold"><i style={{width: pct(p.score)}}/><u style={{left: pct(p.threshold)}}/></div>
            <div className="detail-section"><b>What moved the score</b>
              <p className="sev-note">Compared with a typical recorded crash. Patterns in the data, not causes.</p>
              {contributions.map(c => {
                const val = result.input[FACTOR_INPUT[c.factor] || c.factor];
                return (<div className="sev-bar" key={c.factor}>
                  <span>{c.factor.replace(/ \(.*\)/, '')}{val !== undefined && <em>{String(val)}</em>}</span>
                  <div className="sev-track"><i className={c.effect >= 0 ? 'up' : 'down'} style={{width: `${(Math.abs(c.effect) / maxEffect) * 100}%`}}/></div>
                  <b className={c.effect >= 0 ? 'up' : 'down'}>{c.effect >= 0 ? '+' : ''}{fx(c.effect, 3)}</b></div>);
              })}
              {!result.explanation.contributions && <p className="sev-note">Explanation unavailable (cleaned data file not found).</p>}
              {blank.length > 0 && <p className="sev-note">Not provided, so filled with typical values: {blank.join(', ')}.</p>}
            </div>
          </div>}
        </div>
      </div>

      {analysis && <>
        <div className="sev-charts">
          <div className="table-card sev-analysis"><div className="card-head"><div><div className="eyebrow">FEATURE ABLATION</div><h2>Which inputs matter?</h2></div></div>
            <p className="section-copy">Change in test ROC-AUC when an input group is removed and the model retrained. Orange bars are clearly different from zero; grey bars are within noise.</p>
            <div className="sev-chart"><ResponsiveContainer width="100%" height={250}>
              <BarChart data={ablationData} layout="vertical" margin={{left: 10, right: 20, top: 5, bottom: 5}}>
                <CartesianGrid stroke="#202c3c" horizontal={false}/><XAxis type="number" stroke="#718198" fontSize={10}/>
                <YAxis type="category" dataKey="name" stroke="#9aa7b8" fontSize={10} width={170}/>
                <ReferenceLine x={0} stroke="#8795a8"/>
                <Tooltip cursor={{fill: '#ffffff08'}} content={<ChartTip render={d => <><b>{d.name}</b><br/>Change {fx(d.delta)} (95% CI {d.ci})<br/>{d.clear ? 'Clearly different from 0' : 'Within noise'}</>}/>}/>
                <Bar dataKey="delta" radius={3}>{ablationData.map(d => <Cell key={d.name} fill={d.clear ? '#ff9e48' : '#4b5b70'}/>)}</Bar>
              </BarChart></ResponsiveContainer></div></div>

          <div className="table-card sev-analysis"><div className="card-head"><div><div className="eyebrow">ERROR ANALYSIS</div><h2>Catching severe crashes</h2></div></div>
            <p className="section-copy">{analysis.false_negative_count} of {analysis.severe_count} severe crashes in the test period scored below the threshold. Flagging more crashes catches more severe ones at the cost of false alarms (dashed line = random).</p>
            <div className="sev-chart"><ResponsiveContainer width="100%" height={250}>
              <LineChart data={tradeoff} margin={{left: 0, right: 20, top: 5, bottom: 15}}>
                <CartesianGrid stroke="#202c3c"/><XAxis dataKey="share" type="number" domain={[0, 1]} stroke="#718198" fontSize={10} tickFormatter={pct} label={{value: 'Share of crashes flagged', position: 'insideBottom', offset: -8, fill: '#8795a8', fontSize: 10}}/>
                <YAxis domain={[0, 1]} stroke="#718198" fontSize={10} tickFormatter={pct}/>
                <Tooltip content={<ChartTip render={d => <>Flag {pct(d.share)} of crashes<br/>Severe recall {pct(d.severe)}<br/>Fatal recall {pct(d.fatal)}</>}/>}/>
                <Legend wrapperStyle={{fontSize: 10}}/>
                <Line type="monotone" dataKey="severe" name="Severe recall" stroke="#ff9e48" strokeWidth={2} dot={{r: 2}}/>
                <Line type="monotone" dataKey="fatal" name="Fatal recall" stroke="#ff5964" strokeWidth={2} dot={{r: 2}}/>
                <Line type="linear" dataKey="random" name="Random" stroke="#8795a8" strokeDasharray="4 4" dot={false}/>
              </LineChart></ResponsiveContainer></div></div>
        </div>

        <div className="sev-charts">
          <div className="table-card sev-analysis"><div className="card-head"><div><div className="eyebrow">CALIBRATION</div><h2>Do scores match reality?</h2></div></div>
            <p className="section-copy">Each point is a tenth of the test crashes, ordered by score. On the dashed diagonal, a score of 0.6 would mean 60% of those crashes were severe.</p>
            <div className="sev-chart"><ResponsiveContainer width="100%" height={250}>
              <LineChart data={calib} margin={{left: 0, right: 20, top: 5, bottom: 15}}>
                <CartesianGrid stroke="#202c3c"/><XAxis dataKey="pred" type="number" domain={[0, 1]} stroke="#718198" fontSize={10} label={{value: 'Mean model score', position: 'insideBottom', offset: -8, fill: '#8795a8', fontSize: 10}}/>
                <YAxis domain={[0, 1]} stroke="#718198" fontSize={10} tickFormatter={pct}/>
                <ReferenceLine segment={[{x: 0, y: 0}, {x: 1, y: 1}]} stroke="#8795a8" strokeDasharray="4 4"/>
                <Tooltip content={<ChartTip render={d => <>Score about {fx(d.pred, 2)}<br/>Actually severe {pct(d.obs)}<br/>{d.rows} crashes</>}/>}/>
                <Line type="monotone" dataKey="obs" name="Observed severe rate" stroke="#39c9a5" strokeWidth={2} dot={{r: 3}}/>
              </LineChart></ResponsiveContainer></div></div>

          <div className="table-card sev-analysis"><div className="card-head"><div><div className="eyebrow">MODEL COMPARISON</div><h2>Binary vs four-class</h2></div></div>
            <p className="section-copy">Task A is severe vs non-severe. Task B is four-class severity. Four-class models are collapsed to P(fatal)+P(grievous) to compare on Task A.</p>
            <div className="table-scroll"><table><thead><tr><th>TASK A MODEL</th><th>TEST AUC</th><th>95% CI</th><th>RECALL</th></tr></thead><tbody>
              {cmpA.map(r => <tr key={r.model}><td><b>{short(r.model)}</b></td><td>{fx(r.test_roc_auc)}</td><td>{r.test_roc_auc_95ci}</td><td>{pct(r.severe_recall)}</td></tr>)}
            </tbody></table></div>
            <div className="table-scroll"><table><thead><tr><th>TASK B MODEL</th><th>BAL. ACC.</th><th>MACRO-F1</th><th>FATAL RECALL</th></tr></thead><tbody>
              {cmpB.map(r => <tr key={r.model}><td><b>{short(r.model)}</b></td><td>{fx(r.balanced_accuracy)}</td><td>{fx(r.macro_f1)}</td><td>{pct(r.fatal_recall)}</td></tr>)}
            </tbody></table></div></div>
        </div>

        <details className="table-card sev-analysis sev-details"><summary>Show the ablation and miss-rate tables</summary>
          <div className="table-scroll"><table><thead><tr><th>EXPERIMENT</th><th>TEST AUC</th><th>95% CI</th><th>CHANGE VS FULL</th><th>CLEAR EFFECT?</th></tr></thead><tbody>
            {ablRows.map(r => <tr key={r.model + r.experiment}><td><b>{r.experiment}</b></td><td>{fx(r.test_roc_auc)}</td><td>{r.test_auc_95ci}</td><td>{r.experiment === 'Full model' ? '—' : `${fx(r.delta_vs_full)} (${r.delta_95ci})`}</td><td>{r.experiment === 'Full model' ? '—' : (r.delta_distinguishable_from_0 ? 'yes' : 'no')}</td></tr>)}
          </tbody></table></div>
          <div className="table-scroll"><table><thead><tr><th>TRUE SEVERITY</th><th>CRASHES</th><th>SCORED NON-SEVERE</th><th>SHARE</th></tr></thead><tbody>
            {(analysis.error_by_severity || []).map(r => <tr key={r.true_severity}><td><b>{r.true_severity}</b></td><td>{r.crashes}</td><td>{r.predicted_non_severe}</td><td>{pct(r.share_predicted_non_severe)}</td></tr>)}
          </tbody></table></div>
        </details>
      </>}
      {!analysis && <div className="table-card"><p className="section-copy" style={{padding: 18}}>Run <code>python -m v4_pipeline.severity_analysis</code> to add ablation, calibration and error analysis here.</p></div>}
    </section>
  );
}
