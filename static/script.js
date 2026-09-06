(() => {
  const dropzone = document.getElementById('dropzone');
  const fileInput = document.getElementById('fileInput');
  const loadingState = document.getElementById('loadingState');
  const loadingLabel = document.getElementById('loadingLabel');
  const errorState = document.getElementById('errorState');
  const resultState = document.getElementById('resultState');
  const resetBtn = document.getElementById('resetBtn');

  const specimenImage = document.getElementById('specimenImage');
  const faceNote = document.getElementById('faceNote');
  const verdictLabel = document.getElementById('verdictLabel');
  const confidenceValue = document.getElementById('confidenceValue');
  const gaugeArc = document.getElementById('gaugeArc');
  const realProb = document.getElementById('realProb');
  const fakeProb = document.getElementById('fakeProb');
  const faceDetected = document.getElementById('faceDetected');
  const toggleBtns = document.querySelectorAll('.toggle-btn');

  const GAUGE_LENGTH = 188.5; // matches the arc path length in the SVG
  let lastResult = null;

  const loadingMessages = [
    'Locating face…',
    'Cropping region of interest…',
    'Running texture analysis…',
    'Computing attention map…',
  ];

  function showState(el) {
    [dropzone, loadingState, errorState, resultState].forEach(s => s.hidden = true);
    el.hidden = false;
  }

  function cycleLoadingMessages() {
    let i = 0;
    loadingLabel.textContent = loadingMessages[0];
    return setInterval(() => {
      i = (i + 1) % loadingMessages.length;
      loadingLabel.textContent = loadingMessages[i];
    }, 700);
  }

  dropzone.addEventListener('click', () => fileInput.click());

  ['dragenter', 'dragover'].forEach(evt => {
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.add('dragover');
    });
  });
  ['dragleave', 'drop'].forEach(evt => {
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.remove('dragover');
    });
  });
  dropzone.addEventListener('drop', (e) => {
    const file = e.dataTransfer.files[0];
    if (file) analyzeFile(file);
  });

  fileInput.addEventListener('change', () => {
    if (fileInput.files[0]) analyzeFile(fileInput.files[0]);
  });

  resetBtn.addEventListener('click', () => {
    fileInput.value = '';
    showState(dropzone);
  });

  toggleBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      toggleBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      if (!lastResult) return;
      const view = btn.dataset.view;
      specimenImage.src = view === 'heatmap'
        ? `data:image/png;base64,${lastResult.heatmap_base64}`
        : `data:image/png;base64,${lastResult.original_base64}`;
    });
  });

  async function analyzeFile(file) {
    showState(loadingState);
    const messageTimer = cycleLoadingMessages();

    const formData = new FormData();
    formData.append('image', file);

    try {
      const res = await fetch('/analyze', { method: 'POST', body: formData });
      const data = await res.json();

      clearInterval(messageTimer);

      if (!res.ok) {
        errorState.textContent = data.error || 'Something went wrong analyzing that image.';
        showState(errorState);
        return;
      }

      renderResult(data);
    } catch (err) {
      clearInterval(messageTimer);
      errorState.textContent = 'Could not reach the server. Is app.py still running?';
      showState(errorState);
    }
  }

  function renderResult(data) {
    lastResult = data;

    toggleBtns.forEach(b => b.classList.remove('active'));
    document.querySelector('.toggle-btn[data-view="original"]').classList.add('active');
    specimenImage.src = `data:image/png;base64,${data.original_base64}`;

    const isFake = data.label === 'fake';
    verdictLabel.textContent = isFake ? 'Likely manipulated' : 'Likely authentic';
    verdictLabel.className = 'verdict-label ' + (isFake ? 'is-fake' : 'is-real');

    const confidencePct = Math.round(data.confidence * 100);
    confidenceValue.textContent = `${confidencePct}%`;
    gaugeArc.style.stroke = isFake ? 'var(--fake)' : 'var(--real)';
    const offset = GAUGE_LENGTH * (1 - data.confidence);
    // reset then animate for a clean transition each time
    gaugeArc.style.transition = 'none';
    gaugeArc.style.strokeDashoffset = GAUGE_LENGTH;
    requestAnimationFrame(() => {
      gaugeArc.style.transition = '';
      gaugeArc.style.strokeDashoffset = offset;
    });

    realProb.textContent = `${(data.real_probability * 100).toFixed(1)}%`;
    fakeProb.textContent = `${(data.fake_probability * 100).toFixed(1)}%`;
    faceDetected.textContent = data.face_detected ? 'Yes' : 'No';

    faceNote.textContent = data.face_detected
      ? 'A face was detected and cropped for analysis.'
      : 'No face was detected — the full image was analyzed instead.';

    showState(resultState);
  }
})();
