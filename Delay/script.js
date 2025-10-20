
// script.js: front-end logic for SmartChain demo (no backend required).
// Uses MODEL and SAMPLE_INPUTS provided by data.js

function sigmoid(x){ return 1 / (1 + Math.exp(-x)); }

function createInputFields(){
  const container = document.getElementById('input-form');
  const features = MODEL.features;
  features.forEach(f => {
    const div = document.createElement('div');
    div.className = 'input-group';
    div.innerHTML = `<label for="${f}">${f.replace(/_/g,' ')}</label>
      <input id="${f}" type="number" step="any" />`;
    container.appendChild(div);
  });
}

function readInputs(){
  const features = MODEL.features;
  const vals = features.map((f) => {
    const v = parseFloat(document.getElementById(f).value);
    return isNaN(v) ? 0.0 : v;
  });
  return vals;
}

function normalize(vals){
  return vals.map((v,i) => (v - MODEL.means[i]) / MODEL.scales[i]);
}

function predict(vals){
  const x = normalize(vals);
  const coeff = MODEL.coefficients;
  let linear = MODEL.intercept;
  for(let i=0;i<coeff.length;i++){
    linear += coeff[i] * x[i];
  }
  const prob = sigmoid(linear);
  return prob;
}

function displayResult(prob){
  const probText = document.getElementById('probText');
  const probFill = document.getElementById('probFill');
  const explain = document.getElementById('explain');
  document.getElementById('result').style.display = 'block';
  const pct = Math.round(prob*100);
  probText.innerText = `Delay probability: ${pct}%`;
  probFill.style.width = pct + '%';
  let decision = prob >= 0.5 ? 'High risk of delay — consider intervention.' : 'Low risk of delay.';
  explain.innerText = decision + ' (This is a demo model derived from logistic regression trained on sample data.)';
}

document.getElementById('predictBtn').addEventListener('click', () => {
  const vals = readInputs();
  const prob = predict(vals);
  displayResult(prob);
});

document.getElementById('fillSample').addEventListener('click', () => {
  if(SAMPLE_INPUTS && SAMPLE_INPUTS.length>0){
    const row = SAMPLE_INPUTS[0];
    MODEL.features.forEach(f => {
      document.getElementById(f).value = row[f] ?? 0;
    });
  }
});

// Initialize UI
createInputFields();
document.getElementById('sampleArea').innerText = JSON.stringify(SAMPLE_INPUTS, null, 2);
