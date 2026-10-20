const pendingRunStates=['queued','running','cancelling','interrupted'];
const runStateNames={queued:'等待执行',running:'正在处理',cancelling:'正在停止当前操作',interrupted:'本轮已中断，请结束后重新提问',completed:'讲解已完成',budget_exhausted:'已在预算限制下交付',failed:'本轮未完成',cancelled:'本轮已停止'};
const toolNames={list_materials:'查看资料目录',search_materials:'搜索资料',material_outline:'查看资料导航',read_context:'补读前后文',read_passage:'读取文字',read_page:'读取PDF文字',inspect_page:'查看PDF图像',inspect_media:'查阅音视频片段',recall_learning_context:'核对历史原话',deliver_explanation:'交付讲解'};
let runtimeBinding=0, runtimeSource=null, runtimeTimer=null, selectedRun=null, runtimeCursor=0;
const submissionKey=cid=>'learning-submit:'+cid;
function savedSubmission(cid){try{return JSON.parse(localStorage.getItem(submissionKey(cid))||'null');}catch{return null;}}
function stopRunSubscription(){if(runtimeSource)runtimeSource.close();runtimeSource=null;clearTimeout(runtimeTimer);runtimeTimer=null;}
function runEventLabel(event){
  const p=event.payload||{};
  if(event.type==='tool.started')return '正在'+(toolNames[p.name]||'执行资料工具');
  if(event.type==='tool.finished')return (toolNames[p.name]||'资料工具')+(p.error?'：未取得所需结果':'：已返回')+(p.page?' · 第'+p.page+'页':'');
  if(event.type==='model.started')return p.purpose==='summary'?'正在整理历史摘要':'正在理解问题并组织讲解';
  if(event.type==='model.finished')return p.purpose==='summary'?'摘要请求已返回':'模型已返回，正在处理结果';
  return {'run.accepted':'问题已保存','run.started':'开始处理','run.cancel_requested':'已收到停止请求','run.interrupted':'执行已中断，已有记录保留','run.completed':'本轮讲解已保存','run.failed':'本轮未完成','run.cancelled':'本轮已停止','context.reduced':'已按预算选择历史与图像','summary.finished':'历史整理已结束'}[event.type]||'任务状态已更新';
}
function appendRunEvents(events){
  for(const event of events){if(event.seq<=runtimeCursor)continue;runtimeCursor=event.seq;const li=document.createElement('li');li.textContent=runEventLabel(event);$('runtime-progress').append(li);}
  while($('runtime-progress').children.length>8)$('runtime-progress').firstChild.remove();
}
function paintRun(run){
  selectedRun=run;$('runtime-panel').hidden=false;$('runtime-state').textContent=runStateNames[run.state]||run.state;
  busy=pendingRunStates.includes(run.state);availability();
  $('runtime-cancel').hidden=!run.allowed_actions.includes('cancel');$('runtime-cancel').disabled=run.state==='cancelling';
  $('runtime-cancel').textContent=run.state==='interrupted'?'结束此轮，重新提问':run.state==='queued'?'取消排队':'停止本轮';
  $('runtime-hint').textContent=run.state==='interrupted'?'当前版本保留中断记录，尚不能从中间步骤续跑；结束此轮后可重新提问。':run.state==='cancelling'?'正在结束已开始的操作；已发出的远端请求可能仍会计费。':busy?'刷新或关闭页面不会取消任务。':'';
}
async function refreshRuntime(binding,cid,runId){
  try{
    const events=await api(`/runs/${runId}/events?after=${runtimeCursor}`);
    const run=await api(`/runs/${runId}`);if(binding!==runtimeBinding||cid!==conversation)return;
    appendRunEvents(events);const wasPending=selectedRun&&pendingRunStates.includes(selectedRun.state);paintRun(run);
    if(!pendingRunStates.includes(run.state)){
      stopRunSubscription();if(wasPending)await loadChat();return;
    }
    runtimeTimer=setTimeout(()=>refreshRuntime(binding,cid,runId),1500);
  }catch(error){if(binding!==runtimeBinding)return;$('runtime-hint').textContent='连接中断，正在重新查询任务；后台状态以服务端为准。';runtimeTimer=setTimeout(()=>refreshRuntime(binding,cid,runId),2000);}
}
function subscribeRun(binding,cid,run){
  if(!window.EventSource||run.state==='interrupted'){runtimeTimer=setTimeout(()=>refreshRuntime(binding,cid,run.id),1500);return;}
  const source=new EventSource(`/api/runs/${run.id}/events?after=${runtimeCursor}`);runtimeSource=source;
  source.addEventListener('progress',async message=>{
    if(binding!==runtimeBinding||cid!==conversation)return;
    try{
      const event=JSON.parse(message.data);appendRunEvents([event]);
      if(['run.completed','run.failed','run.cancelled','run.interrupted'].includes(event.type)){
        source.close();await refreshRuntime(binding,cid,run.id);
      }else if(event.type==='run.started')paintRun({...selectedRun,state:'running'});
      else if(event.type==='run.cancel_requested')paintRun({...selectedRun,state:'cancelling',allowed_actions:[]});
    }catch(error){notice(error.message,true);}
  });
  source.onerror=()=>{source.close();if(binding!==runtimeBinding)return;$('runtime-hint').textContent='进度连接中断，正在重新查询；任务继续运行。';runtimeTimer=setTimeout(()=>refreshRuntime(binding,cid,run.id),1000);};
}
async function bindRuntime(){
  const binding=++runtimeBinding,cid=conversation;stopRunSubscription();selectedRun=null;runtimeCursor=0;$('runtime-progress').replaceChildren();
  $('runtime-panel').hidden=!cid;$('runtime-retry-submit').hidden=!savedSubmission(cid);if(!cid){busy=false;availability();return;}
  busy=true;availability();
  try{
    const runs=await api(`/conversations/${cid}/runs?limit=1`);if(binding!==runtimeBinding||cid!==conversation)return;
    if(!runs.length){busy=false;$('runtime-panel').hidden=!savedSubmission(cid);$('runtime-state').textContent=savedSubmission(cid)?'上次提交尚未确认，请重试确认。':'';$('runtime-cancel').hidden=true;availability();return;}
    const run=runs[0];paintRun(run);
    const events=await api(`/runs/${run.id}/events?after=${Math.max(0,run.last_event_seq-8)}`);if(binding!==runtimeBinding||cid!==conversation)return;
    runtimeCursor=Math.max(0,run.last_event_seq-8);appendRunEvents(events);
    if(pendingRunStates.includes(run.state))subscribeRun(binding,cid,run);
  }catch(error){if(binding!==runtimeBinding)return;$('runtime-state').textContent='无法确认当前任务，请稍后重试';$('runtime-hint').textContent=error.message;runtimeTimer=setTimeout(bindRuntime,2000);}
}
async function sendRuntimePayload(cid,payload){
  const response=await api(`/conversations/${cid}/runs`,jsonOptions('POST',payload));
  if(savedSubmission(cid)?.client_request_id===payload.client_request_id)localStorage.removeItem(submissionKey(cid));
  if(cid!==conversation)return;
  if($('question').value.trim()===payload.content)$('question').value='';clearFeedback();
  await loadChat();notice('问题已保存，可查看进度或停止本轮。');return response;
}
async function submitRuntimeQuestion(){
  if(busy||!conversation)return;const cid=conversation;
  const body={content:$('question').value.trim(),focus,feedback:feedbackDraft?{...feedbackDraft,focus_text:$('feedback-focus').value}:null};
  const old=savedSubmission(cid), identity=JSON.stringify(body);
  const payload=old&&JSON.stringify({content:old.content,focus:old.focus,feedback:old.feedback})===identity?old:{...body,client_request_id:crypto.randomUUID()};
  localStorage.setItem(submissionKey(cid),JSON.stringify(payload));busy=true;availability();notice('正在提交问题…');
  try{await sendRuntimePayload(cid,payload);}catch(error){if(cid!==conversation)return;notice(error.message+'；已保留问题，可重试确认上次提交。',true);await bindRuntime();}
}
$('runtime-cancel').onclick=guarded(async()=>{
  const cid=conversation,run=selectedRun;if(!run)return;const result=await api(`/runs/${run.id}/cancel`,{method:'POST'});
  if(cid!==conversation)return;paintRun(result);if(!pendingRunStates.includes(result.state))await loadChat();
});
$('runtime-retry-submit').onclick=guarded(async()=>{
  const cid=conversation,payload=savedSubmission(cid);if(!payload)return;
  await sendRuntimePayload(cid,payload);
});
