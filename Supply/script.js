
import DATA from './data.js';

const logEl = document.getElementById('log');
const statusEl = document.getElementById('status');
const trainBtn = document.getElementById('trainBtn');
const predictBtn = document.getElementById('predictBtn');
const ctx = document.getElementById('chart').getContext('2d');

function log(msg){
  logEl.textContent += msg + '\n';
  logEl.scrollTop = logEl.scrollHeight;
}

// Prepare tensors from DATA
const sequences = DATA.sequences;
const targets = DATA.targets;
const windowSize = DATA.window_size;
const nFeatures = DATA.feature_names.length;

const X = tf.tensor3d(sequences); // [samples, timesteps, features]
const y = tf.tensor2d(targets.map(t=>[t])); // [samples,1]

// simple train/val split
const split = Math.floor(X.shape[0] * 0.8);
const X_train = X.slice([0,0,0],[split,windowSize,nFeatures]);
const X_val   = X.slice([split,0,0],[X.shape[0]-split,windowSize,nFeatures]);
const y_train = y.slice([0,0],[split,1]);
const y_val   = y.slice([split,0],[y.shape[0]-split,1]);

let model = null;

function buildModel(){
  const mdl = tf.sequential();
  mdl.add(tf.layers.lstm({units:64, returnSequences:false, inputShape:[windowSize,nFeatures]}));
  mdl.add(tf.layers.dropout({rate:0.2}));
  mdl.add(tf.layers.dense({units:32, activation:'relu'}));
  mdl.add(tf.layers.dense({units:1, activation:'sigmoid'})); // sigmoid because targets normalized 0-1
  mdl.compile({optimizer: tf.train.adam(0.001), loss:'meanSquaredError', metrics:['mse']});
  return mdl;
}

let chart = null;
function createChart(trainLoss=[], valLoss=[]){
  const data = {
    labels: trainLoss.map((_,i)=>i+1),
    datasets: [
      { label:'train loss', data: trainLoss, tension:0.2 },
      { label:'val loss', data: valLoss, tension:0.2 }
    ]
  };
  if(chart) chart.destroy();
  chart = new Chart(ctx, {
    type:'line',
    data,
    options:{ responsive:true, maintainAspectRatio:false }
  });
}

trainBtn.addEventListener('click', async ()=>{
  statusEl.textContent = 'Building model...';
  model = buildModel();
  statusEl.textContent = 'Training...';
  trainBtn.disabled = true;
  const history = await model.fit(X_train, y_train, {
    epochs: 30,
    batchSize: 32,
    validationData: [X_val, y_val],
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        const t = `Epoch ${epoch+1}: loss=${logs.loss.toFixed(4)} val_loss=${logs.val_loss.toFixed(4)}`;
        log(t);
        createChart(
          model._trainLogs?model._trainLogs.map(l=>l.loss):[],
          model._trainLogs?model._trainLogs.map(l=>l.val_loss):[]
        );
      }
    }
  });
  statusEl.textContent = 'Training complete.';
  predictBtn.disabled = false;
});

predictBtn.addEventListener('click', async ()=>{
  if(!model){ alert('Train the model first'); return; }
  statusEl.textContent = 'Running predictions on validation set...';
  // Predict on validation set
  const preds = model.predict(X_val);
  const predsArr = await preds.data();
  // Un-normalize
  const y_min = DATA.scaler.y_min;
  const y_max = DATA.scaler.y_max;
  const y_range = (y_max - y_min) || 1.0;
  const unNorm = (v) => v * y_range + y_min;
  const actualTensor = y_val;
  const actualArrNorm = await actualTensor.data();
  const actualArr = Array.from(actualArrNorm).map(unNorm);
  const predArr = Array.from(predsArr).map(unNorm);

  // display first 30 pairs
  log('\nFirst 30 predictions (actual vs predicted):');
  for(let i=0;i<Math.min(30,predArr.length);i++){
    log(`${i+1}. actual=${actualArr[i].toFixed(2)}  pred=${predArr[i].toFixed(2)}`);
  }
  statusEl.textContent = 'Done predictions.';
});

// helpful cleanup when page unloads
window.addEventListener('unload', ()=> {
  if(model) model.dispose();
  X.dispose(); y.dispose();
});
