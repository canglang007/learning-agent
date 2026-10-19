let memorySettings=null, memoryTopics=[];
const memoryKinds={preference:'讲解偏好',goal:'学习目标',unresolved:'待解释问题'};
const jsonOptions=(method,body)=>({method,headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
async function loadMemory(){
  const selected=conversation;$('memory-panel').hidden=!selected;if(!selected)return;
  const [settings,topics,materials,messages]=await Promise.all([
    api(`/conversations/${selected}/context-settings`),api(`/spaces/${space}/learning-topics`),
    api(`/spaces/${space}/materials`),api(`/conversations/${selected}/messages`)
  ]);
  if(selected!==conversation)return;
  memorySettings=settings;memoryTopics=topics;
  $('memory-topic').innerHTML='<option value="">不选择主题</option>'+topics.map(t=>`<option value="${t.id}">${esc(t.title)} · ${t.material_ids.length}份资料 · ${t.id.slice(0,6)}</option>`).join('');
  $('memory-topic').value=settings.topic_id||'';$('memory-enabled').checked=!!settings.memory_enabled;
  $('topic-materials').innerHTML=materials.map(m=>`<option value="${m.id}">${esc(m.name)}</option>`).join('');
  $('memory-source').innerHTML='<option value="">由我直接填写</option>'+messages.filter(m=>m.role==='assistant'&&m.run_id).slice(-12).map(m=>`<option value="${m.run_id}">${esc(m.content.slice(0,65))}</option>`).join('');
  await showMemories();
}
async function showMemories(){
  const selected=conversation, tid=$('memory-topic').value, topic=memoryTopics.find(t=>t.id===tid);
  const savedTopic=memoryTopics.find(t=>t.id===memorySettings?.topic_id);
  $('memory-state').textContent=(memorySettings?.memory_enabled?`此会话已接续“${savedTopic?.title||'所选主题'}”。每轮按范围选取相关条目，允许没有匹配。`:'此会话独立学习，未使用跨会话记忆。')+(tid!==(memorySettings?.topic_id||'')?' 当前查看的主题尚未应用到此会话。':'');
  $('memory-items').replaceChildren();$('memory-material').innerHTML='<option value="">整个主题</option>';
  $('memory-form').hidden=!tid;
  if(!topic)return;
  const materials=await api(`/spaces/${space}/materials`), names=new Map(materials.map(m=>[m.id,m.name]));
  $('memory-material').innerHTML+=topic.material_ids.map(id=>`<option value="${id}">${esc(names.get(id)||id)}</option>`).join('');
  const items=await api(`/spaces/${space}/memories?topic_id=${encodeURIComponent(tid)}`);
  if(selected!==conversation||tid!==$('memory-topic').value)return;
  if(!items.length)$('memory-items').textContent='此主题还没有保存的记忆。';
  for(const m of items){
    const row=document.createElement('article'), p=document.createElement('p');
    const stale=Object.entries(m.material_revisions||{}).some(([id,rev])=>materials.find(x=>x.id===id)?.revision!==rev);
    p.textContent=`${memoryKinds[m.kind]} · ${m.status==='active'?'启用':'停用'}${stale?' · 资料已更新，暂停召回；核对后修改保存以重新确认':''} · ${m.origin==='user_confirmed'?'你确认的记录':'你直接填写'} · 修订${m.revision}`;row.append(p);
    const body=document.createElement('p');body.textContent=m.text;row.append(body);
    const scope=document.createElement('small');scope.textContent='范围：'+(m.material_ids.length?m.material_ids.map(id=>names.get(id)||id).join('、'):'整个主题');row.append(scope);
    if(m.source){const a=document.createElement('a');a.href='#'+new URLSearchParams({conversation:m.source.conversation_id,answer:m.source.run_id});a.textContent=' 查看原回答 ';a.onclick=guarded(async e=>{e.preventDefault();if(busy)throw Error('请等待本轮回答完成后再查看其他会话');if(location.hash===a.hash)await openHash();else location.hash=a.hash;});row.append(a);}
    const editor=document.createElement('textarea');editor.value=m.text;editor.maxLength=600;editor.rows=2;editor.setAttribute('aria-label','修改记忆内容');row.append(editor);
    for(const [label,method,patch] of [
      ['保存修改','PATCH',()=>({revision:m.revision,text:editor.value.trim()})],
      [m.status==='active'?'停用':'重新启用','PATCH',()=>({revision:m.revision,status:m.status==='active'?'disabled':'active'})],
      ['遗忘此记忆','DELETE',()=>null]
    ]){
      const b=document.createElement('button');b.type='button';b.textContent=label;b.disabled=busy;
      b.onclick=guarded(async()=>{
        await api(`/spaces/${space}/memories/${m.id}`+(method==='DELETE'?`?revision=${m.revision}`:''),method==='DELETE'?{method}:jsonOptions(method,patch()));
        await loadLearning();await loadMemory();notice('记忆已更新；旧自动上下文已失效，原始对话保留。');
      });row.append(b);
    }
    $('memory-items').append(row);
  }
  availability();
}
$('memory-topic').onchange=guarded(showMemories);
$('context-settings-form').onsubmit=guarded(async e=>{
  e.preventDefault();await api(`/conversations/${conversation}/context-settings`,jsonOptions('PATCH',{
    revision:memorySettings.revision,topic_id:$('memory-topic').value||null,memory_enabled:$('memory-enabled').checked
  }));await loadLearning();await loadMemory();notice('接续设置已保存，之后的新会话仍默认独立学习。');
});
$('topic-form').onsubmit=guarded(async e=>{
  e.preventDefault();const t=await api(`/spaces/${space}/learning-topics`,jsonOptions('POST',{
    title:$('topic-title').value.trim(),material_ids:[...$('topic-materials').selectedOptions].map(o=>o.value)
  }));$('topic-title').value='';await loadMemory();$('memory-topic').value=t.id;await showMemories();notice('主题已创建，请保存接续设置以关联此会话。');
});
$('memory-form').onsubmit=guarded(async e=>{
  e.preventDefault();await api(`/spaces/${space}/memories`,jsonOptions('POST',{
    topic_id:$('memory-topic').value,kind:$('memory-kind').value,text:$('memory-text').value.trim(),
    material_ids:$('memory-material').value?[$('memory-material').value]:[],source_run_id:$('memory-source').value||null
  }));$('memory-text').value='';await loadLearning();await loadMemory();notice('已保存；只有选择接续此主题的会话才会使用。');
});
