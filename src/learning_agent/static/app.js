const $ = id => document.getElementById(id);
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const clock = s => s>=3600?`${Math.floor(s/3600)}:${String(Math.floor(s/60)%60).padStart(2,'0')}:${String(Math.floor(s%60)).padStart(2,'0')}`:`${Math.floor(s/60)}:${String(Math.floor(s%60)).padStart(2,'0')}`;
let indexPoll=null, jobsPoll=null, jobsSignature='', outlineMaterial='', feedbackDraft=null, feedbackStarter='';
const feedbackNames={stuck:'这一步没懂',example:'换个例子',restate:'我来复述',understood:'这一点明白了'};
const approachNames={clarify:'澄清卡点',prerequisite:'补前置知识',step_by_step:'拆开推导',example:'数值例子',analogy:'类比',source_check:'核对资料',check_restatement:'检查复述',explain:'解释概念',acknowledge:'确认反馈'};
let space='', conversation='', configured=false, busy=false, focus=null, media=null, poll=null, asr='disabled';
let uploading=false, pauseUpload=false;
let learningRevision=0;
async function api(path, options={}) {
  const response=await fetch('/api'+path,options), text=await response.text();
  if(response.status===401&&path!=='/me'){location.href='/login';throw Error('登录已过期，正在跳转登录页');}
  let data=null;
  if(text){try{data=JSON.parse(text);}catch{data=text;}}
  if(!response.ok) throw Error(typeof data?.detail==='string'?data.detail:JSON.stringify(data?.detail??data));
  return data;
}
const post=(path,content)=>api(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({content})});
function notice(text,error=false) {$('notice').textContent=text;$('notice').className=error?'error':'';}
function guarded(fn) {return async e=>{try{await fn(e);}catch(error){notice(error.message,true);}};}
function availability() {
  $('send').disabled=busy;
  $('spaces').disabled=uploading;
  document.querySelectorAll('#space-form input,#space-form button,#upload-form input,#upload-form button').forEach(el=>el.disabled=busy);
  if(uploading){$('spaces').disabled=true;document.querySelectorAll('#space-form input,#space-form button,#upload-form input,#upload-form button').forEach(el=>el.disabled=true);}
  $('send').disabled=busy||!conversation||!configured;
  document.querySelectorAll('.feedback-actions button,#cancel-feedback').forEach(el=>el.disabled=busy);
  document.querySelectorAll('#learning-editor button,#learning-questions button,#reset-learning,#memory-panel input,#memory-panel select,#memory-panel textarea,#memory-panel button').forEach(el=>el.disabled=busy);
}
function markdown(node,text) {
  // Protect TeX before Markdown consumes backslashes; leave code spans/blocks untouched.
  const token='MATH'+crypto.randomUUID().replaceAll('-',''), equations=[];
  const protectedText=text.replace(/```[\s\S]*?```|`[^`\n]*`|\$\$[\s\S]*?\$\$|\\\[[\s\S]*?\\\]|\\\([\s\S]*?\\\)|\$[^$\n]+\$/g, value=>{
    if(value.startsWith('`')) return value;
    equations.push(value);return token+(equations.length-1)+'ENDMATH';
  });
  node.innerHTML=DOMPurify.sanitize(marked.parse(protectedText),{USE_PROFILES:{html:true},FORBID_TAGS:['img','video','audio','iframe','style']});
  const walker=document.createTreeWalker(node,NodeFilter.SHOW_TEXT);
  const texts=[];while(walker.nextNode())texts.push(walker.currentNode);
  for(const t of texts)t.textContent=t.textContent.replace(new RegExp(token+'(\\d+)ENDMATH','g'),(_,n)=>equations[Number(n)]);
  renderMathInElement(node,{delimiters:[{left:'$$',right:'$$',display:true},{left:'\\[',right:'\\]',display:true},{left:'\\(',right:'\\)',display:false},{left:'$',right:'$',display:false}],throwOnError:false,trust:false,strict:'warn'});
}
function player() {return media?.has_video?$('video-player'):$('audio-player');}
const CODEC_PROBES={av1:'video/mp4; codecs="av01.0.05M.08"',h264:'video/mp4; codecs="avc1.42E01E"',vp9:'video/webm; codecs="vp9"',vp8:'video/webm; codecs="vp8"'};
function browserCanPlay(codec) {
  // 服务端只能按容器+编码判断"理论上能播"；真的解不解得了要问浏览器自己。
  if(!codec||!CODEC_PROBES[codec]) return true;
  return document.createElement('video').canPlayType(CODEC_PROBES[codec])!=='';
}
async function ensureBrowserPlayable(info) {
  if(info.playback!=='direct'||!info.has_video||browserCanPlay(info.video_codec)||info.transcodeRequested) return;
  info.transcodeRequested=true;
  notice(`当前浏览器解不了 ${String(info.video_codec).toUpperCase()}，正在转成 H.264……`);
  await api(`/spaces/${space}/materials/${info.material_id}/prepare?force=1`,{method:'POST'});
  poll=setTimeout(()=>refreshMedia().catch(e=>notice(e.message,true)),2000);
}
function linkReferences(body,references) {
  const verified=references.filter(r=>r.position_verified);
  const walker=document.createTreeWalker(body,NodeFilter.SHOW_TEXT), texts=[];
  while(walker.nextNode())if(!walker.currentNode.parentElement.closest('a,code,pre,.katex'))texts.push(walker.currentNode);
  for(const text of texts){
    const matches=[];
    for(const ref of verified){let from=0,index;while((index=text.textContent.indexOf(ref.label,from))!==-1){matches.push({index,ref});from=index+ref.label.length;}}
    if(!matches.length)continue;
    matches.sort((a,b)=>a.index-b.index);const fragment=document.createDocumentFragment();let cursor=0;
    for(const {index,ref} of matches){if(index<cursor)continue;fragment.append(document.createTextNode(text.textContent.slice(cursor,index)));const a=document.createElement('a');a.href=ref.url;a.textContent=ref.label;
      a.onclick=guarded(async e=>{e.preventDefault();if(ref.kind==='media')await showMedia(ref.material_id,ref.start_seconds,ref.end_seconds);else await showPage(ref.material_id,ref.page);});
      fragment.append(a);cursor=index+ref.label.length;
    }
    fragment.append(document.createTextNode(text.textContent.slice(cursor)));text.replaceWith(fragment);
  }
}
function clearPosition() {
  focus=null;$('reader').hidden=true;$('media-reader').hidden=true;$('focus-hint').textContent='';
  $('video-player').pause();$('audio-player').pause();media=null;clearTimeout(poll);
  $('video-player').ontimeupdate=null;$('audio-player').ontimeupdate=null;
}
async function loadSpaces() {
  const items=await api('/spaces');$('spaces').innerHTML=items.map(s=>`<option value="${s.id}">${esc(s.name)}</option>`).join('');
  space=items.find(s=>s.id===localStorage.getItem('learning-space'))?.id||items[0]?.id||'';$('spaces').value=space;await loadSpace(true);
}
async function loadSpace(reset=false) {
  if(reset){clearFeedback();clearPosition();$('navigation').hidden=true;}localStorage.setItem('learning-space',space);
  const items=space?await api(`/spaces/${space}/materials`):[];
  $('materials').innerHTML=items.map(m=>`<li><button type="button" data-material="${m.id}">${esc(m.name)}${m.format==='pdf'?' · '+m.page_count+'页':m.media_state?' · '+({processing:'准备中',ready:'可回放',failed:'准备失败'}[m.media_state]||m.media_state):''}</button><button type="button" data-outline="${m.id}">查看目录</button></li>`).join('');
  const conversations=space?await api(`/spaces/${space}/conversations`):[];
  $('conversations').innerHTML=conversations.map(c=>`<option value="${c.id}">${esc(c.title||'新会话 '+c.id.slice(0,8))}</option>`).join('');
  await refreshIndex();
  await refreshJobs();
  refreshUploadRecovery();
  conversation=conversations.find(c=>c.id===conversation)?.id||conversations[0]?.id||'';
  if(!conversation&&space){const created=await api(`/spaces/${space}/conversations`,{method:'POST'});$('conversations').innerHTML=`<option value="${created.id}">新会话</option>`;conversation=created.id;}
  $('conversations').value=conversation;await loadChat();
}
async function loadChat() {
  const selectedConversation=conversation;
  const messages=selectedConversation?await api(`/conversations/${selectedConversation}/messages`):[];
  if(selectedConversation!==conversation)return;
  $('chat').replaceChildren();
  $('starter-questions').hidden=!selectedConversation||Boolean(messages.length);
  if(!messages.length)$('chat').innerHTML='<p>创建会话后提出困惑。你可以直接追问，不必按固定学习流程操作。</p>';
  for(const m of messages) {
    const article=document.createElement('article');article.innerHTML=`<small>${m.role==='user'?'你的问题':m.kind==='system'?'系统状态':'讲解'}</small><div class="body"></div>`;
    const body=article.querySelector('.body');if(m.role==='assistant'&&m.run_id)article.id='answer-'+m.run_id;
    if(m.role==='assistant'&&m.kind!=='system')markdown(body,m.content);else{body.classList.add('user');body.textContent=m.content;}
    if(m.role==='assistant'&&m.run_id) {
      const sources=document.createElement('div');sources.className='sources';article.append(sources);
      const details=document.createElement('details');details.innerHTML='<summary>查看本轮工具与执行状态</summary><pre>加载中…</pre>';article.append(details);
      api(`/runs/${m.run_id}`).then(r=>{
        if(selectedConversation!==conversation)return;
        linkReferences(body,r.references||[]);
        if((r.context||[]).some(step=>step.omissions?.length)) {
          const hint=document.createElement('p');hint.textContent='本次仅使用部分历史或图像。原记录仍保留；需要早期内容时，请指定回答或资料位置。';
          article.insertBefore(hint,details);
        }
        if(['completed','budget_exhausted'].includes(r.state))addFeedbackActions(article,m.run_id,body);
        details.querySelector('pre').textContent=JSON.stringify(r,null,2);const seen=new Set();
        for(const event of r.events) {
          const o=event.observation;if(event.type!=='tool'||!o||o.error)continue;
          if(o.source_url&&!seen.has(o.source_url)) {
            seen.add(o.source_url);const button=document.createElement('button');button.textContent=`回听 ${o.name} ${clock(o.start_seconds)}–${clock(o.end_seconds)}`;
            button.onclick=guarded(()=>showMedia(o.material_id,o.start_seconds,o.end_seconds));sources.append(button);
          }else if(o.page_url&&!seen.has(o.page_url)) {
            seen.add(o.page_url);const button=document.createElement('button');button.textContent=`查看 ${o.name} 第${o.page}页`;
            button.onclick=guarded(()=>showPage(o.material_id,o.page));sources.append(button);
          }
        }
      }).catch(e=>{details.querySelector('pre').textContent=e.message;});
    }
    $('chat').append(article);
  }
  await loadLearning();await loadMemory();await bindRuntime();availability();
}
async function showPage(id,page) {
  const base=`/spaces/${space}/materials/${id}`, data=await api(`${base}/pages/${page}`), pages=await api(`${base}/pages`);
  clearPosition();focus={material_id:id,page};$('reader').hidden=false;$('reader-title').textContent=`${data.name} · 第${page}/${pages.length}页`;
  $('page-number').value=page;$('page-number').max=pages.length;$('page-image').src='/api'+base+`/pages/${page}/image`;$('open-page').href=$('page-image').src;
  $('page-text').textContent=data.text||'本页没有可提取文字，Agent 可查看页面图像。';$('focus-hint').textContent=`当前选中：${data.name} 第${page}页（Agent仍需自行查阅）`;
  $('reader').scrollIntoView({behavior:'smooth',block:'start'});
}
function selectClip() {
  if(!media)return;
  const start=Number($('clip-start').value), end=Number($('clip-end').value);
  if(!(start>=0&&end>start&&end<=media.duration&&end-start<=120))throw Error('请选择时长内有效片段，每段最多120秒');
  focus={material_id:media.material_id,start_seconds:start,end_seconds:end};
  $('focus-hint').textContent=`当前选中：${media.name} ${clock(start)}–${clock(end)}（Agent仍需自行查阅）`;
  return focus;
}
async function showMedia(id,start=0,end=null) {
  const info=await api(`/spaces/${space}/materials/${id}/media`);
  clearPosition();media=info;
  const material=await api(`/spaces/${space}/materials/${id}`), source=JSON.parse(material.source||'{}');
  $('source-url').value=source.url||'';$('source-start').value=source.start_seconds||0;
  $('origin-link').hidden=!source.url;
  if(source.url){
    const url=new URL(source.url), originalStart=(source.start_seconds||0)+start;
    if(['www.youtube.com','youtube.com','youtu.be'].includes(url.hostname))url.searchParams.set('t',Math.floor(originalStart)+'s');
    $('origin-link').href=url.href;$('origin-link').textContent='打开原始来源 · '+clock(originalStart);
  }
  $('media-reader').hidden=false;$('media-title').textContent=`${info.name} · ${clock(info.duration)}`;
  $('clip-start').value=start;$('clip-end').value=end??Math.min(info.duration,start+30);
  if(info.state==='ready') {
    const p=player();p.hidden=false;p.src=info.playback_url;p.onloadedmetadata=()=>{p.currentTime=start;};
    (info.has_video?$('audio-player'):$('video-player')).hidden=true;
    await ensureBrowserPlayable(info).catch(e=>notice(e.message,true));
  }else{$('video-player').hidden=true;$('audio-player').hidden=true;}
  selectClip();await refreshMedia();$('media-reader').scrollIntoView({behavior:'smooth',block:'start'});
}
async function refreshMedia() {
  clearTimeout(poll);
  if(!media)return;
  const id=media.material_id, info=await api(`/spaces/${space}/materials/${id}/media`);
  if(media?.material_id!==id)return;
  const wasReady=media.state==='ready', wasTranscribing=media.transcript_state==='running';media=info;
  const materialButton=$('materials').querySelector(`[data-material="${id}"]`);
  if(materialButton)materialButton.textContent=info.name+' · '+({ready:'可回放',processing:'准备中',failed:'准备失败'}[info.state]||info.state);
  $('media-state').textContent=`回放：${{ready:'就绪',processing:'等待或正在准备',failed:'失败',paused:'已暂停',cancelled:'已取消'}[info.state]||info.state}${info.error?' · '+info.error:''}；字幕：${{ready:'已就绪',missing:'未提供',running:'等待或正在转写',silent:'未识别到讲话，可上传字幕',failed:'转写失败',paused:'已暂停',cancelled:'已取消'}[info.transcript_state]||info.transcript_state}${info.transcript_error?' · '+info.transcript_error:''}${info.segment_count&&info.transcript_state!=='ready'?'；已有字幕仍可使用':''}`;
  $('retry-media').hidden=info.state!=='failed';$('transcribe').disabled=asr==='disabled'||!info.has_audio||info.transcript_state==='running';
  $('caption-file').disabled=false;$('caption-form').querySelector('button').disabled=false;
  $('asr-hint').textContent=asr==='disabled'?'未配置ASR：可以上传字幕。无字幕的视频仍可看图，但不能解释未识别的讲话。':`转写方式：${asr}。仅点击后运行；云端转写会上传音频并可能计费。`;
  $('caption-source').textContent=(info.transcript_source||'没有时间字幕')+(info.alignment_notes?.length?'；有'+info.alignment_notes.length+'处时间对齐说明（窗口裁剪、分片跨界或重复合并），明细见工具记录。':'');
  const segments=await api(`/spaces/${space}/materials/${id}/segments`);
  if(media?.material_id!==id)return;
  $('segments').replaceChildren();
  for(const s of segments){const b=document.createElement('button');b.textContent=`${clock(s.start)}–${clock(s.end)} ${s.text}`;b.onclick=guarded(()=>{ $('clip-start').value=s.start;$('clip-end').value=Math.min(s.end,s.start+120);selectClip();seek(); });$('segments').append(b);}
  if(!wasReady&&info.state==='ready'){const p=player();p.hidden=false;p.src=info.playback_url;await ensureBrowserPlayable(info).catch(e=>notice(e.message,true));}
  if(wasTranscribing&&info.transcript_state!=='running')await refreshIndex();
  if(info.state==='processing'||info.transcript_state==='running')poll=setTimeout(()=>refreshMedia().catch(e=>notice(e.message,true)),2000);
}
function seek() {
  selectClip();const p=player();p.currentTime=focus.start_seconds;
  const stop=focus.end_seconds;p.ontimeupdate=()=>{if(p.currentTime>=stop)p.pause();};
  p.play().catch(e=>notice('请点击播放器继续播放：'+e.message,true));
}
$('spaces').onchange=guarded(async()=>{space=$('spaces').value;await loadSpace(true);});
$('conversations').onchange=guarded(async()=>{clearFeedback();conversation=$('conversations').value;await loadChat();});
$('starter-questions').onclick=e=>{const text=e.target.dataset.starter;if(!text)return;$('question').value=text;$('question').focus();};
$('space-form').onsubmit=guarded(async e=>{e.preventDefault();const item=await post('/spaces',$('space-name').value);localStorage.setItem('learning-space',item.id);await loadSpaces();notice('资料空间已创建');});
$('upload-form').onsubmit=guarded(async e=>{
  e.preventDefault();if(!space)throw Error('请先创建资料空间');
  const file=$('file').files[0];if(!file)return;
  if(/\.(mp4|mov|webm|mkv|mp3|m4a|wav|ogg|flac)$/i.test(file.name)){
    if(localStorage.getItem(uploadKey())){refreshUploadRecovery();notice('有未完成的媒体上传。请确认所选为同一原文件后继续，或先取消旧上传。');return;}
    const session=await api(`/spaces/${space}/uploads`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({filename:file.name,size:file.size,last_modified:file.lastModified})});
    localStorage.setItem(uploadKey(),JSON.stringify({id:session.id,name:file.name,size:file.size,lastModified:file.lastModified}));
    return sendMedia(file,session);
  }
  notice('正在上传并检查资料…');const data=new FormData();data.append('file',$('file').files[0]);const item=await api(`/spaces/${space}/materials`,{method:'POST',body:data});
  await loadSpace();if(item.material_id)await showMedia(item.material_id);notice(item.material_id?'音视频已收录，正在后台准备回放。':'资料已就绪');
});
const uploadKey=()=>`learning-upload-${space}`;
function pendingUpload(){try{return JSON.parse(localStorage.getItem(uploadKey())||'null');}catch{return null;}}
function refreshUploadRecovery(){
  const pending=pendingUpload();$('upload-recovery').hidden=!pending||uploading;
  if(!pending)return;
  $('upload-recovery-hint').textContent=`未完成：${pending.name}。刷新后请重新选择同一原文件；名称、大小和修改时间只能辅助匹配，请你确认内容相同。`;
  const file=$('file').files[0];$('resume-upload').disabled=uploading||!file||file.name!==pending.name||file.size!==pending.size||file.lastModified!==pending.lastModified;
}
async function sendMedia(file,session){
  uploading=true;pauseUpload=false;availability();refreshUploadRecovery();$('pause-upload').hidden=false;$('pause-upload').disabled=false;
  try{
    while(session.state==='uploading'&&session.received<session.size){
      if(pauseUpload){notice('上传已暂停。已确认文件块会保留，可继续上传。');return;}
      const end=Math.min(session.size,session.received+session.chunk_size);
      session=await api(`/uploads/${session.id}`,{method:'PATCH',headers:{'Upload-Offset':String(session.received)},body:file.slice(session.received,end)});
      $('upload-progress').textContent=`已确认 ${Math.floor(session.received/session.size*100)}%（${session.received}/${session.size} 字节）`;
    }
    if(pauseUpload){notice('文件块已保存，继续时将完成媒体检查。');return;}
    $('upload-progress').textContent='文件已上传，正在检查媒体并完成收录…';
    const item=await api(`/uploads/${session.id}/complete`,{method:'POST'});
    localStorage.removeItem(uploadKey());$('upload-progress').textContent='媒体上传完成';
    await loadSpace();await showMedia(item.id);notice('整份媒体已收录，回放由后台处理；转写需另外点击开始。');
  }finally{uploading=false;$('pause-upload').hidden=true;availability();refreshUploadRecovery();}
}
$('file').onchange=refreshUploadRecovery;
$('pause-upload').onclick=()=>{pauseUpload=true;$('pause-upload').disabled=true;};
$('resume-upload').onclick=guarded(async()=>{
  const pending=pendingUpload(), file=$('file').files[0];
  if(!pending||!file||file.name!==pending.name||file.size!==pending.size||file.lastModified!==pending.lastModified)throw Error('请重新选择同一原文件，或取消旧上传后重新开始');
  const session=await api(`/uploads/${pending.id}`);
  if(session.state==='cancelled'){localStorage.removeItem(uploadKey());refreshUploadRecovery();throw Error('此上传已取消，请重新上传');}
  await sendMedia(file,session);
});
$('cancel-upload').onclick=guarded(async()=>{const pending=pendingUpload();if(!pending||uploading)return;await api(`/uploads/${pending.id}`,{method:'DELETE'});localStorage.removeItem(uploadKey());refreshUploadRecovery();$('upload-progress').textContent='临时上传已取消';});
$('new-conversation').onclick=guarded(async()=>{clearFeedback();if(!space)throw Error('请先创建资料空间');const c=await api(`/spaces/${space}/conversations`,{method:'POST'});await loadSpace();conversation=c.id;$('conversations').value=c.id;await loadChat();});
$('delete-conversation').onclick=guarded(async()=>{clearFeedback();if(!conversation)throw Error('当前没有可删除的会话');if(!confirm('删除这个会话和它的全部问答记录？删除后无法恢复。'))return;await api(`/conversations/${conversation}`,{method:'DELETE'});conversation='';await loadSpace();notice('会话已删除');});
$('ask-form').onsubmit=async e=>{e.preventDefault();await submitRuntimeQuestion();};
$('materials').onclick=guarded(async e=>{
  if(e.target.dataset.outline)return showOutline(e.target.dataset.outline);
  const id=e.target.dataset.material;if(!id)return;const m=await api(`/spaces/${space}/materials/${id}`);
  if(m.format==='pdf')return showPage(id,1);
  if(['video','audio'].includes(m.format))return showMedia(id);
  const d=document.createElement('details');d.open=true;const s=document.createElement('summary');s.textContent=m.name;const p=document.createElement('pre');p.textContent=m.text;d.append(s,p);$('chat').prepend(d);
});
$('page-form').onsubmit=guarded(e=>{e.preventDefault();if(focus?.page)return showPage(focus.material_id,Number($('page-number').value));});
$('close-reader').onclick=$('close-media').onclick=clearPosition;
$('select-clip').onclick=guarded(selectClip);$('play-clip').onclick=guarded(seek);
$('select-current').onclick=guarded(()=>{const t=player().currentTime;$('clip-start').value=Math.max(0,t-5).toFixed(3);$('clip-end').value=Math.min(media.duration,t+25).toFixed(3);selectClip();});
$('caption-form').onsubmit=guarded(async e=>{e.preventDefault();const data=new FormData();data.append('file',$('caption-file').files[0]);await api(`/spaces/${space}/materials/${media.material_id}/captions`,{method:'POST',body:data});await refreshMedia();await refreshIndex();notice('时间字幕已保存，请重新准备语义检索。');});
$('transcribe').onclick=guarded(async()=>{await api(`/spaces/${space}/materials/${media.material_id}/transcribe`,{method:'POST'});await refreshMedia();});
$('retry-media').onclick=guarded(async()=>{await api(`/spaces/${space}/materials/${media.material_id}/prepare`,{method:'POST'});await refreshMedia();});
async function refreshIndex() {
  clearTimeout(indexPoll);if(!space){$('index-status').textContent='';return;}
  const selected=space, status=await api(`/spaces/${space}/retrieval`);if(selected!==space)return;
  $('index-status').textContent=`段落 ${status.chunks} · 向量 ${status.embedded}/${status.chunks} · ${status.dense_ready?'语义检索就绪':status.provider==='disabled'?'未配置向量，使用关键词检索':({queued:'等待执行',running:'正在准备',paused:'已暂停',cancelled:'已取消',failed:'准备失败',pending:'待准备'}[status.state]||'待准备')}${status.error?' · '+status.error:''}`;
  $('prepare-index').disabled=status.provider==='disabled'||['running','queued','paused'].includes(status.state)||busy;
  if(['running','queued'].includes(status.state))indexPoll=setTimeout(()=>refreshIndex().catch(e=>notice(e.message,true)),2000);
}
async function refreshJobs() {
  clearTimeout(jobsPoll);if(!space){jobsSignature='';$('jobs').replaceChildren();$('worker-state').textContent='请先创建资料空间';return;}
  const selected=space, result=await api(`/spaces/${space}/jobs`);if(selected!==space)return;
  const signature=selected+JSON.stringify(result);
  if(signature===jobsSignature){jobsPoll=setTimeout(()=>refreshJobs().catch(e=>notice(e.message,true)),3000);return;}
  const listed=await api(`/spaces/${selected}/materials`);if(selected!==space)return;
  for(const m of listed){const button=$('materials').querySelector(`[data-material="${m.id}"]`);if(button&&m.media_state)button.textContent=m.name+' · '+({processing:'准备中',ready:'可回放',failed:'准备失败',paused:'已暂停',cancelled:'已取消'}[m.media_state]||m.media_state);}
  jobsSignature=signature;
  $('worker-state').textContent=result.worker_online?'处理进程在线':'处理进程离线；任务会保留。请在项目目录另开终端运行：uv run python -m learning_agent.worker';
  $('jobs').replaceChildren();
  const names={preview:'准备回放',transcribe:'语音转写',index:'准备语义检索'}, states={queued:'等待执行',running:'处理中',paused:'已暂停',failed:'失败',cancelled:'已取消',succeeded:'完成'};
  for(const job of result.jobs.slice(0,15)) {
    const row=document.createElement('div'), title=document.createElement('small');
    const name=job.material_id?$('materials').querySelector(`[data-material="${job.material_id}"]`)?.textContent?.split(' · ')[0]:'';
    title.textContent=`${name?name+' · ':''}${names[job.kind]}：${states[job.state]}${job.pause_requested&&job.state==='running'?'，等待当前步骤结束':''}${job.kind==='index'&&job.state==='running'?'，段落 '+job.completed+'/'+job.total:''}${job.kind==='transcribe'&&job.total_seconds?'，已保存 '+clock(job.processed_seconds)+' / '+clock(job.total_seconds)+'，分片 '+job.completed+'/'+job.total:''}${job.error?' · '+job.error:''}`;row.append(title);
    const actions=job.state==='queued'||job.state==='running'?['pause','cancel']:job.state==='paused'?['resume','cancel']:job.state==='failed'?['retry','cancel']:[];
    for(const action of actions){const b=document.createElement('button');b.type='button';b.textContent=({pause:'暂停',cancel:'取消',resume:'继续',retry:job.error?.includes('remote_result_unknown')?'知晓可能再次计费后重试':'重试'})[action];b.disabled=action==='pause'&&Boolean(job.pause_requested);
      b.onclick=guarded(async()=>{b.disabled=true;await api(`/jobs/${job.id}/${action}`,{method:'POST'});await refreshJobs();await refreshMedia();await refreshIndex();});row.append(b);}
    $('jobs').append(row);
  }
  if(!result.jobs.length)$('jobs').textContent='还没有后台任务';
  jobsPoll=setTimeout(()=>refreshJobs().catch(e=>notice(e.message,true)),3000);
}
async function showOutline(id,start=0) {
  const selected=space, result=await api(`/spaces/${space}/materials/${id}/outline?start=${start}`);if(selected!==space)return;
  outlineMaterial=id;$('navigation').hidden=false;$('navigation').open=true;$('navigation-title').textContent=result.name+' · 目录';
  if(!start)$('navigation-entries').replaceChildren();
  for(const entry of result.entries){const button=document.createElement('button');button.type='button';button.textContent=(entry.page?'第'+entry.page+'页':entry.start_seconds!==undefined?clock(entry.start_seconds)+'–'+clock(entry.end_seconds):'第'+entry.start_line+'行')+' '+entry.label;
    button.onclick=guarded(async()=>{if(entry.page)await showPage(id,entry.page);else if(entry.start_seconds!==undefined)await showMedia(id,entry.start_seconds,Math.min(entry.end_seconds,entry.start_seconds+120));else{$('materials').querySelector(`[data-material="${id}"]`).click();}});$('navigation-entries').append(button);}
  $('navigation-more').hidden=result.next_start===null;$('navigation-more').onclick=guarded(()=>showOutline(outlineMaterial,result.next_start));
}
$('prepare-index').onclick=guarded(async()=>{if(!space)return;$('prepare-index').disabled=true;await api(`/spaces/${space}/retrieval`,{method:'POST'});notice('正在准备语义检索，期间可继续使用关键词检索。');await refreshIndex();});
$('source-form').onsubmit=guarded(async e=>{
  e.preventDefault();const id=media.material_id, selected={...focus};
  await api(`/spaces/${space}/materials/${id}/source`,{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify({url:$('source-url').value||null,start_seconds:Number($('source-start').value)})});
  if(media?.material_id===id)await showMedia(id,selected.start_seconds,selected.end_seconds);
  await refreshIndex();notice('来源已保存，索引需要重新准备。');
});
function clearFeedback() {
  if(feedbackStarter&&$('question').value===feedbackStarter)$('question').value='';feedbackStarter='';
  feedbackDraft=null;$('feedback-draft').hidden=true;$('feedback-focus').value='';
}
function addFeedbackActions(article,runId,body) {
  const actions=document.createElement('div');actions.className='feedback-actions';
  for(const [kind,label] of Object.entries(feedbackNames)){
    const button=document.createElement('button');button.type='button';button.textContent=label;button.disabled=busy;
    button.onclick=()=>{
      const selection=window.getSelection(), text=selection&&body.contains(selection.anchorNode)&&body.contains(selection.focusNode)?selection.toString().slice(0,1000):'';
      feedbackDraft={kind,target_run_id:runId};$('feedback-draft').hidden=false;
      $('feedback-label').textContent=label+' · ';const target=document.createElement('a');target.href='#answer-'+runId;target.textContent='针对这次讲解（返回查看）';$('feedback-label').append(target,'，发送后才记录。');$('feedback-focus').value=text;
      const starters={stuck:'这一步我还没懂：',example:'请围绕这个卡点换一个具体例子：',restate:'我的理解是：',understood:'这一点明白了。'};
      if(!$('question').value.trim()||$('question').value===feedbackStarter){feedbackStarter=starters[kind];$('question').value=feedbackStarter;}
      $('question').focus();$('ask-form').scrollIntoView({behavior:'smooth',block:'start'});
    };actions.append(button);
  }
  article.append(actions);
}
async function loadLearning() {
  const selected=conversation;$('learning-panel').hidden=!selected;if(!selected)return;
  const state=await api(`/conversations/${selected}/learning`);if(selected!==conversation)return;
  const note=state.working_note, current=state.state;learningRevision=state.revision;$('learning-note').replaceChildren();
  const facts=[['goal','学习目标'],['active_topic','当前主题'],['confusion','当前卡点']];
  for(const [key,label] of facts){
    const fact=current[key], p=document.createElement('p');p.textContent=`${label}：${fact?.text||'尚未明确'}${fact?'（'+(key==='goal'&&current.goal_pinned?'你固定的目标':fact.origin==='user_explicit'?'你明确填写':'Agent工作判断，可修正')+'）':''}`;$('learning-note').append(p);
    $('learning-'+key).value=fact?.text||'';
  }
  $('learning-goal-pinned').checked=!!current.goal_pinned;
  for(const text of note?[`当前讲法：${approachNames[note.approach]||note.approach}；${note.reason}`]:['可以直接提问，或修改当前学习目标。']){
    const p=document.createElement('p');p.textContent=text;$('learning-note').append(p);
  }
  $('learning-attempts').textContent=state.attempts.length?'最近讲解尝试：'+state.attempts.map(a=>approachNames[a.approach]||a.approach).join(' → '):'';
  $('learning-feedback').replaceChildren();
  for(const f of state.feedback.slice(-4)){
    const p=document.createElement('p');p.textContent=`已提交反馈：${feedbackNames[f.kind]}${f.focus_text?'；卡点：'+f.focus_text:''} `;
    const link=document.createElement('a');link.href='#answer-'+f.target_run_id;link.textContent='返回对应讲解';p.append(link);$('learning-feedback').append(p);
  }
  $('learning-questions').replaceChildren();
  for(const q of current.open_questions){
    const p=document.createElement('p');p.textContent=`${q.status==='open'?'待解释':q.status==='user_understood'?'你已反馈理解':'你已关闭'}：${q.text} `;
    if(q.status==='open'){
      const close=document.createElement('button');close.type='button';close.textContent='关闭此问题';close.style.width='auto';
      close.onclick=guarded(async()=>{await editLearning({close_question_ids:[q.id]});});p.append(close);
    }
    if(q.source_run_id){const a=document.createElement('a');a.href='#answer-'+q.source_run_id;a.textContent='来源回答';p.append(a);}
    $('learning-questions').append(p);
  }
  $('learning-summary').replaceChildren();
  if(state.summary){
    const title=document.createElement('p');title.textContent=`历史摘要（到第${state.summary.covered_end}轮，仅保留所选原话要点；可核对来源）`;$('learning-summary').append(title);
    for(const item of state.summary.payload.items){
      const p=document.createElement('p');p.textContent=item.text+' ';
      for(const id of item.source_run_ids){const a=document.createElement('a');a.href='#answer-'+id;a.textContent='来源回合 ';p.append(a);}
      $('learning-summary').append(p);
    }
  }
}
async function editLearning(patch){
  await api(`/conversations/${conversation}/learning`,{method:'PATCH',headers:{'Content-Type':'application/json'},body:JSON.stringify({revision:learningRevision,...patch})});
  await loadLearning();notice('学习状态已保存，旧摘要已失效；原始对话仍保留。');
}
$('learning-edit-form').onsubmit=guarded(async e=>{e.preventDefault();await editLearning({
  goal:$('learning-goal').value.trim()||null,goal_pinned:$('learning-goal-pinned').checked,active_topic:$('learning-active_topic').value.trim()||null,
  confusion:$('learning-confusion').value.trim()||null,add_questions:$('learning-add-question').value.trim()?[$('learning-add-question').value.trim()]:[]
});$('learning-add-question').value='';});
$('reset-learning').onclick=guarded(async()=>{await editLearning({reset:true});});
$('cancel-feedback').onclick=clearFeedback;
async function openHash() {
  const p=new URLSearchParams(location.hash.slice(1)), requestedConversation=p.get('conversation'), id=p.get('media');
  if(requestedConversation){
    for(const s of await api('/spaces')){
      if((await api(`/spaces/${s.id}/conversations`)).some(c=>c.id===requestedConversation)){
        clearFeedback();space=s.id;$('spaces').value=space;conversation=requestedConversation;await loadSpace(true);$('learning-panel').open=true;if(p.get('answer'))$('answer-'+p.get('answer'))?.scrollIntoView();return;
      }
    }
    notice('当前本机资料空间没有此会话',true);return;
  }
  if(!id)return;
  for(const s of await api('/spaces')){
    if((await api(`/spaces/${s.id}/materials`)).some(m=>m.id===id)){
      space=s.id;$('spaces').value=space;await loadSpace(true);await showMedia(id,Number(p.get('start')||0),p.has('end')?Number(p.get('end')):null);return;
    }
  }
  notice('当前本机资料空间没有此媒体',true);
}
window.onhashchange=guarded(openHash);
(async()=>{try{const me=await api('/me');if(me?.authenticated){$('account-name').textContent=me.username;$('account-bar').hidden=false;}const s=await api('/status');configured=s.configured;asr=s.asr_provider;const c=s.capabilities||{};const modes=[c.text?'文本':'无文本',c.vision?'视觉':'无视觉',c.tool_calling?'工具':'无工具'];$('status').textContent=configured?'模型已配置，可以直接提问。':'未配置模型：可以先浏览资料或已有对话；提问前运行 ./start.sh --configure。';$('tech-status-body').textContent=`模型：${s.model} · 协议：${s.protocol} · 能力：${modes.join(' / ')} · 媒体上限 ${(s.media_max_bytes/1024**3).toFixed(2)} GiB / ${(s.media_max_seconds/3600).toFixed(1)} 小时`;$('tech-status').hidden=false;await loadSpaces();await openHash();}catch(e){notice(e.message,true);}})();
