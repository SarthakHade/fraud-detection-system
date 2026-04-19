/* ============================================================
   UI Core - Cursors & Navigation
   ============================================================ */
const cursor=document.getElementById('cursor'),ring=document.getElementById('cursor-ring');
let mx=0,my=0,rx=0,ry=0;
document.addEventListener('mousemove',e=>{mx=e.clientX;my=e.clientY;cursor.style.left=mx+'px';cursor.style.top=my+'px'});
function animRing(){rx+=(mx-rx)*0.12;ry+=(my-ry)*0.12;ring.style.left=rx+'px';ring.style.top=ry+'px';requestAnimationFrame(animRing)}
animRing();

/* ============================================================
   Background Hex Canvas Animation
   ============================================================ */
(function(){
  const c=document.getElementById('bg-canvas');
  if(!c) return;
  const ctx=c.getContext('2d');
  let W,H,hexes=[]; const S=40,R=S/Math.sqrt(3);
  function init(){
    W=c.width=window.innerWidth; H=c.height=window.innerHeight; hexes=[];
    const colW=S*1.5+2, rowH=R*2;
    for(let r=0;r<Math.ceil(H/rowH)+2;r++){
      for(let cc=0;cc<Math.ceil(W/colW)+2;cc++){
        hexes.push({x:cc*(S*1.5+2)+(r%2?S*0.75:0), y:r*R, phase:Math.random()*Math.PI*2, speed:0.2+Math.random()*0.3});
      }
    }
  }
  function draw(){
    ctx.clearRect(0,0,W,H);
    let t=Date.now()*0.001;
    ctx.lineWidth=0.5;
    hexes.forEach(h=>{
      let b = 0.2 + 0.8*Math.abs(Math.sin(t*h.speed+h.phase));
      ctx.strokeStyle=`rgba(34,211,238,${0.02 + b*0.06})`;
      ctx.beginPath();
      for(let i=0;i<6;i++){
        let a = i*Math.PI/3 - Math.PI/6;
        ctx.lineTo(h.x + (R-2)*Math.cos(a), h.y + (R-2)*Math.sin(a));
      }
      ctx.closePath(); ctx.stroke();
    });
    requestAnimationFrame(draw);
  }
  window.addEventListener('resize',init); init(); draw();
})();

/* ============================================================
   Tab Switching & UI Initializations
   ============================================================ */
function switchTab(id, btn){
  document.querySelectorAll('.tab-btn').forEach(b=>b.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(c=>c.classList.remove('active'));
  btn.classList.add('active');
  document.getElementById(id).classList.add('active');
  // Lazy-init Anomalies scatter the first time that tab is opened
  if(id === 'tab-an' && !window._scatterBuilt){
    window._scatterBuilt = true;
    setTimeout(buildScatter, 50); // tiny delay so element is painted
  }
  // Animate Explainability bars the first time that tab is opened
  if(id === 'tab-ex' && !window._barsAnimated){
    window._barsAnimated = true;
    setTimeout(()=>{
      document.querySelectorAll('.fac-bar-fill').forEach(b=>b.style.width=b.dataset.w+'%');
    }, 100);
  }
}

function buildScatter(){
  // Use fixed 800×350 coordinate space (matches the viewBox in HTML)
  const SW = 760, SH = 310, PAD = 40;
  const svg = document.getElementById('scatter-svg');
  if(!svg) return;

  for(let i = 0; i < 120; i++){
    let isA = Math.random() > 0.78;
    // Anomalies: top-left cluster (high amount, low freq). Safe: bottom-right
    let cx = isA
      ? PAD + 20 + Math.random() * 220
      : PAD + 180 + Math.random() * (SW - PAD - 200);
    let cy = isA
      ? 20 + Math.random() * 120
      : 140 + Math.random() * (SH - 140);
    
    const c = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    c.setAttribute('cx', cx);
    c.setAttribute('cy', cy);
    c.setAttribute('r', isA ? 6 : 4);
    c.setAttribute('fill', isA ? 'var(--red)' : 'var(--cyan)');
    c.setAttribute('opacity', isA ? '0.9' : '0.6');
    c.setAttribute('class', 'scatter-dot');
    if(isA) c.setAttribute('filter', 'drop-shadow(0 0 4px rgba(248,113,113,0.8))');

    c.addEventListener('mouseover', (e) => {
      const tt = document.getElementById('ano-tt');
      const box = document.getElementById('scatter-box').getBoundingClientRect();
      tt.textContent = isA ? `⚠ High-Value Anomaly` : `✓ Standard Transaction`;
      tt.style.opacity = '1';
      tt.style.left = (e.clientX - box.left + 12) + 'px';
      tt.style.top  = (e.clientY - box.top  - 36) + 'px';
    });
    c.addEventListener('mouseout', () => document.getElementById('ano-tt').style.opacity = '0');
    c.addEventListener('click', () => {
      document.getElementById('ano-panel').style.display = 'block';
      const id  = 'TX-' + Math.floor(1000 + Math.random() * 8000);
      const v   = isA ? 'Unknown Shell Corp' : 'Verified Vendor';
      const amt = Math.floor(10000 + Math.random() * 900000);
      document.getElementById('ap-id').textContent  = id;
      document.getElementById('ap-v').textContent   = v;
      document.getElementById('ap-amt').textContent = '₹' + amt.toLocaleString();
      document.getElementById('ap-rsn').textContent = isA
        ? 'Extreme outlier on Amount axis. Distance metric 3.7σ above cluster centroid.'
        : 'Within normal operational variance bounds.';
    });
    svg.appendChild(c);
  }
}

// Tooltips for Trends
window.showTT = function(e, t){
  let tt = document.getElementById('tt-line');
  tt.textContent = "Data Segment Overlap";
  tt.style.opacity = '1';
  tt.style.left = (e.offsetX+10)+'px';
  tt.style.top = (e.offsetY+10)+'px';
}
window.hideTT = function(){
  document.getElementById('tt-line') && (document.getElementById('tt-line').style.opacity='0');
}
