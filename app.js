// Configs globales
const CAM_COUNT = 4;
let model; // tfjs model
let cams = [];
let recorders = [];
let motionBuffers = []; // pour différence d'images
let immobileTimers = []; // compte à rebours par caméra
let zones = []; // masques polygonaux
let config = {
  freeMobile: { user: "", pass: "" },
  scoreMin: 0.35,
  motionThresh: 12,
  countdownSec: 30
};

// Chargement config et modèle
(async function init() {
  try {
    const cfg = await fetch('config.json').then(r => r.json());
    Object.assign(config, cfg);
  } catch {}
  document.getElementById('scoreMin').value = config.scoreMin;
  document.getElementById('motionThresh').value = config.motionThresh;
  document.getElementById('countdownSec').value = config.countdownSec;

  // Charger modèle YOLOv11 converti en TF.js
  // Format attendu: ./modeles/model.json + shards
  model = await tf.loadGraphModel('./modeles/model.json'); // Export TF.js recommandé pour exécuter YOLO11 dans le navigateur

  bindUI();
})();

function bindUI() {
  document.getElementById('start').onclick = startAll;
  document.getElementById('stop').onclick = stopAll;
  document.querySelectorAll('.editZone').forEach(btn => btn.onclick = () => editZone(+btn.dataset.idx));
  document.getElementById('scoreMin').onchange = e => (config.scoreMin = +e.target.value);
  document.getElementById('motionThresh').onchange = e => (config.motionThresh = +e.target.value);
  document.getElementById('countdownSec').onchange = e => (config.countdownSec = +e.target.value);
}

async function startAll() {
  document.getElementById('start').disabled = true;
  document.getElementById('stop').disabled = false;
  for (let i = 0; i < CAM_COUNT; i++) await startCam(i);
}

async function startCam(idx) {
  const video = document.getElementById('video' + idx);
  const overlay = document.getElementById('overlay' + idx);
  const stateEl = document.getElementById('state' + idx);
  const countEl = document.getElementById('count' + idx);

  // Demander la webcam (différents devices: sélectionner via constraints)
  const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 }, audio: false });
  video.srcObject = stream;
  overlay.width = 640; overlay.height = 480;

  // Enregistreur vidéo
  const recorder = new MediaRecorder(stream, { mimeType: 'video/webm;codecs=vp9' });
  let chunks = [];
  recorder.ondataavailable = e => chunks.push(e.data);
  recorder.onstop = () => saveClip(idx, new Blob(chunks, { type: 'video/webm' }));
  recorders[idx] = recorder;

  motionBuffers[idx] = null;
  immobileTimers[idx] = null;

  // Boucle d’inférence
  const ctx = overlay.getContext('2d');
  (function loop() {
    if (document.getElementById('stop').disabled) return; // arrêté
    tf.engine().startScope();
    drawFrame(video, ctx);
    inferAndDetect(idx, video, ctx, stateEl, countEl);
    tf.engine().endScope();
    requestAnimationFrame(loop);
  })();
}

function stopAll() {
  document.getElementById('start').disabled = false;
  document.getElementById('stop').disabled = true;
  for (let i = 0; i < CAM_COUNT; i++) {
    const v = document.getElementById('video' + i);
    v.srcObject && v.srcObject.getTracks().forEach(t => t.stop());
    recorders[i] && recorders[i].state !== 'inactive' && recorders[i].stop();
    clearCountdown(i);
  }
}

function drawFrame(video, ctx) {
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  ctx.drawImage(video, 0, 0, ctx.canvas.width, ctx.canvas.height);
}

function tensorFromVideo(video) {
  return tf.tidy(() => {
    const input = tf.browser.fromPixels(video).toFloat();
    const resized = tf.image.resizeBilinear(input, [640, 640]);
    const normalized = resized.div(255.0);
    return normalized.expandDims(0);
  });
}

// Post-processing YOLO: dépend du graphe exporté. Ici, on suppose sortie [boxes, scores, classes] déjà décodée.
// Sinon, implémenter décodage + NMS (cf. libs TF.js YOLO pour guidance).
async function inferAndDetect(idx, video, ctx, stateEl, countEl) {
  const input = tensorFromVideo(video);
  const outputs = await model.executeAsync(input); // peut être tensor unique ou liste
  const detections = await parseDetections(outputs); // [{x,y,w,h,score,cls}]
  tf.dispose([input, outputs]);

  // Filtrer classe "person"
  const persons = detections.filter(d => d.cls === 'person' && d.score >= config.scoreMin);

  // Appliquer masque de zone si défini
  const zone = zones[idx];
  const inZone = (d) => !zone || polygonIntersectsBox(zone, d);

  const candidates = persons.filter(inZone);

  // Dessin
  candidates.forEach(d => drawBox(ctx, d, '#22c55e'));
  // Déterminer posture
  const fallLike = candidates.some(isLyingDown);
  const sittingLike = candidates.some(isSitting);
  const inBedZone = candidates.some(d => isInExcludedBedZone(idx, d));

  const motion = measureMotion(idx, ctx);
  const immobile = motion < config.motionThresh;

  // État logique
  if (fallLike && !sittingLike && !inBedZone) {
    stateEl.textContent = 'Personne au sol';
    startCountdown(idx, stateEl, countEl, immobile);
    // démarrer enregistrement si pas déjà
    if (recorders[idx].state === 'inactive') {
      recorders[idx].start(1000);
    }
  } else {
    stateEl.textContent = sittingLike ? 'Assis(e)' : 'OK';
    clearCountdown(idx);
    if (recorders[idx].state !== 'inactive') recorders[idx].stop();
    countEl.textContent = '—';
  }
}

function parseDetections(outputs) {
  // Adapter selon votre export TF.js.
  // Exemple générique: outputs: { boxes: [N,4], scores: [N], classes: [N] }
  return Promise.resolve([]); // TODO: implémenter selon votre modèle
}

function drawBox(ctx, d, color = '#3b82f6') {
  ctx.strokeStyle = color; ctx.lineWidth = 2;
  ctx.strokeRect(d.x, d.y, d.w, d.h);
  ctx.fillStyle = color;
  ctx.fillText(`${d.cls} ${(d.score*100).toFixed(0)}%`, d.x+4, d.y+12);
}

function isLyingDown(d) {
  const ratio = d.w / d.h; // horizontale => ratio élevé
  const low = d.y + d.h > (0.75 * 480); // proche du bas
  return ratio > 1.4 || low;
}

function isSitting(d) {
  const ratio = d.h / d.w; // verticale
  return ratio > 1.2;
}

function isInExcludedBedZone(idx, d) {
  // si une zone "bed" est définie, exclure
  return false; // à adapter selon vos zones
}

function polygonIntersectsBox(poly, d) {
  // simple test: centroid dans le polygone
  const cx = d.x + d.w/2, cy = d.y + d.h/2;
  return pointInPolygon(cx, cy, poly);
}
function pointInPolygon(x, y, poly) {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const xi = poly[i][0], yi = poly[i][1];
    const xj = poly[j][0], yj = poly[j][1];
    const intersect = ((yi > y) !== (yj > y)) &&
      (x < (xj - xi) * (y - yi) / ((yj - yi) || 1e-6) + xi);
    if (intersect) inside = !inside;
  }
  return inside;
}

function measureMotion(idx, ctx) {
  const frame = ctx.getImageData(0, 0, ctx.canvas.width, ctx.canvas.height);
  if (!motionBuffers[idx]) {
    motionBuffers[idx] = frame;
    return 0;
  }
  // différence d’images (approximation)
  let diff = 0;
  const a = frame.data, b = motionBuffers[idx].data;
  for (let i = 0; i < a.length; i += 4) {
    diff += Math.abs(a[i] - b[i]) + Math.abs(a[i+1] - b[i+1]) + Math.abs(a[i+2] - b[i+2]);
  }
  motionBuffers[idx] = frame;
  return diff / (ctx.canvas.width * ctx.canvas.height);
}

function startCountdown(idx, stateEl, countEl, immobile) {
  if (immobileTimers[idx]) {
    // mettre à jour affichage
    countEl.textContent = immobileTimers[idx].remaining + 's';
    return;
  }
  let remaining = config.countdownSec;
  countEl.textContent = remaining + 's';
  immobileTimers[idx] = { remaining, interval: setInterval(() => {
    // si mouvement > seuil, annuler
    if (!immobile) {
      clearCountdown(idx);
      stateEl.textContent = 'Mouvement détecté';
      countEl.textContent = '—';
      return;
    }
    remaining -= 1;
    immobileTimers[idx].remaining = remaining;
    countEl.textContent = remaining + 's';
    if (remaining <= 0) {
      clearCountdown(idx);
      triggerAlert(idx);
    }
  }, 1000) };
}

function clearCountdown(idx) {
  const t = immobileTimers[idx];
  if (t) { clearInterval(t.interval); immobileTimers[idx] = null; }
}

async function triggerAlert(idx) {
  const msg = encodeURIComponent(`Alerte chute détectée • Caméra ${idx+1} • ${new Date().toLocaleString()}`);
  const { user, pass } = config.freeMobile;
  if (!user || !pass) return log(`Alerte cam ${idx+1}: SMS non envoyé (identifiants manquants)`, 'warn');

  try {
    // API Free Mobile SMS
    const url = `https://smsapi.free-mobile.fr/sendmsg?user=${encodeURIComponent(user)}&pass=${encodeURIComponent(pass)}&msg=${msg}`;
    const res = await fetch(url);
    if (res.ok) {
      log(`SMS envoyé • Cam ${idx+1}`, 'ok');
    } else {
      log(`Échec SMS (${res.status}) • Cam ${idx+1}`, 'warn');
    }
  } catch (e) {
    log(`Erreur SMS: ${e.message} • Cam ${idx+1}`, 'warn');
  }

  // capture screenshot
  const canvas = document.getElementById('overlay' + idx);
  canvas.toBlob(blob => saveCapture(idx, blob), 'image/png');

  // assurer enregistrement 30s supplémentaires
  if (recorders[idx].state === 'inactive') recorders[idx].start(1000);
  setTimeout(() => { if (recorders[idx].state !== 'inactive') recorders[idx].stop(); }, 30000);

  // mettre en avant le stream fautif (UI)
  focusCamera(idx);
}

function saveCapture(idx, blob) {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `captures/camera${idx+1}_${Date.now()}.png`;
  a.click();
}

function saveClip(idx, blob) {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `captures/camera${idx+1}_${Date.now()}.webm`;
  a.click();
}

function log(text, status = '') {
  const li = document.createElement('li');
  li.textContent = text;
  if (status) li.classList.add(status);
  document.getElementById('log').prepend(li);
}

function editZone(idx) {
  // UI pour dessiner un polygone sur la vidéo (à implémenter).
  alert('Éditeur de zone à implémenter (polygone)');
}

function focusCamera(idx) {
  // Optionnel: déplacer la section de la caméra fautive en haut, agrandir
  const grid = document.getElementById('grid');
  const section = document.querySelector(`section.cam[data-idx="${idx}"]`);
  grid.prepend(section);
}
