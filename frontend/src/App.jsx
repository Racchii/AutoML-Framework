import { useState } from 'react';
import { Upload, Activity, Layers, Play } from 'lucide-react';
import './App.css';

function App() {
  const [file, setFile] = useState(null);
  const [taskType, setTaskType] = useState('classification');
  const [targetColumn, setTargetColumn] = useState('');
  const [schema, setSchema] = useState(null);
  const [datasetId, setDatasetId] = useState(null);
  const [useTuning, setUseTuning] = useState(false);
  const [tuningTrials, setTuningTrials] = useState(10);
  const [isTraining, setIsTraining] = useState(false);
  const [results, setResults] = useState(null);

  const handleFileChange = (e) => {
    if (e.target.files[0]) {
      setFile(e.target.files[0]);
    }
  };

  const handleUpload = async () => {
    if (!file) return;
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
      // In a real app, use the actual backend URL
      const response = await fetch('http://localhost:8000/upload', {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      setSchema(data.schema);
      setDatasetId(data.dataset_id);
      
      // Auto-select last column as target usually
      if (data.schema.columns.length > 0) {
        setTargetColumn(data.schema.columns[data.schema.columns.length - 1]);
      }
    } catch (error) {
      console.error('Error uploading file:', error);
      alert('Failed to upload dataset.');
    }
  };

  const handleTrain = async () => {
    if (!datasetId || !targetColumn) return;
    
    setIsTraining(true);
    const formData = new FormData();
    formData.append('target_column', targetColumn);
    formData.append('task_type', taskType);
    formData.append('models', 'random_forest,xgboost'); // hardcoded for now
    formData.append('use_tuning', useTuning);
    formData.append('tuning_trials', tuningTrials);
    
    try {
      const response = await fetch(`http://localhost:8000/train/${datasetId}`, {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      
      // Fetch results directly (assuming synchronous for this prototype)
      const resResponse = await fetch(`http://localhost:8000/results/${data.job_id}`);
      const resData = await resResponse.json();
      setResults(resData.metrics);
    } catch (error) {
      console.error('Error training:', error);
      alert('Failed to train models.');
    } finally {
      setIsTraining(false);
    }
  };

  return (
    <div className="app-container animate-fade-in">
      <header>
        <h1 className="header-title">AutoML Nexus</h1>
        <p className="header-subtitle">Intelligent Machine Learning Automation</p>
      </header>

      {!schema && (
        <div className="glass-panel" style={{ textAlign: 'center', padding: '60px 20px' }}>
          <Layers size={48} color="var(--primary-color)" style={{ marginBottom: '20px' }} />
          <h2 style={{ marginBottom: '16px' }}>Upload Your Dataset</h2>
          <p style={{ color: 'var(--text-secondary)', marginBottom: '32px' }}>
            Get started by uploading a CSV file. We'll automatically infer the schema and prepare it for modeling.
          </p>
          
          <div style={{ display: 'flex', justifyContent: 'center', gap: '16px', alignItems: 'center' }}>
            <input 
              type="file" 
              accept=".csv" 
              onChange={handleFileChange} 
              className="input-field"
              style={{ maxWidth: '300px' }}
            />
            <button 
              className="btn btn-primary" 
              onClick={handleUpload}
              disabled={!file}
            >
              <Upload size={18} />
              Process Data
            </button>
          </div>
        </div>
      )}

      {schema && !results && (
        <div className="glass-panel animate-fade-in">
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '24px' }}>
            <Activity size={24} color="var(--secondary-color)" />
            <h2>Configure Training Task</h2>
          </div>
          
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '24px', marginBottom: '32px' }}>
            <div>
              <label className="label">Target Column</label>
              <select 
                className="input-field" 
                value={targetColumn} 
                onChange={(e) => setTargetColumn(e.target.value)}
              >
                {schema.columns.map(col => (
                  <option key={col} value={col}>{col}</option>
                ))}
              </select>
            </div>
            
            <div>
              <label className="label">Task Type</label>
              <select 
                className="input-field" 
                value={taskType} 
                onChange={(e) => setTaskType(e.target.value)}
              >
                <option value="classification">Classification</option>
                <option value="regression">Regression</option>
              </select>
            </div>
          </div>
          
          <div style={{ background: 'rgba(255,255,255,0.03)', padding: '16px', borderRadius: '8px', marginBottom: '32px' }}>
            <h3 style={{ fontSize: '1.1rem', marginBottom: '16px' }}>Advanced Settings</h3>
            <div style={{ display: 'flex', alignItems: 'center', gap: '24px' }}>
              <label style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer' }}>
                <input 
                  type="checkbox" 
                  checked={useTuning} 
                  onChange={(e) => setUseTuning(e.target.checked)} 
                  style={{ width: '18px', height: '18px', accentColor: 'var(--primary-color)' }}
                />
                Enable Hyperparameter Tuning (Optuna)
              </label>
              
              {useTuning && (
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <label className="label" style={{ marginBottom: 0 }}>Trials:</label>
                  <input 
                    type="number" 
                    className="input-field" 
                    style={{ width: '100px', padding: '8px' }}
                    value={tuningTrials}
                    onChange={(e) => setTuningTrials(parseInt(e.target.value))}
                    min="1"
                    max="100"
                  />
                </div>
              )}
            </div>
          </div>
          
          <button 
            className="btn btn-primary" 
            onClick={handleTrain}
            disabled={isTraining}
            style={{ width: '100%', padding: '16px', fontSize: '1.1rem' }}
          >
            {isTraining ? 'Training Models...' : (
              <>
                <Play size={20} />
                Start AutoML Pipeline
              </>
            )}
          </button>
        </div>
      )}

      {results && (
        <div className="glass-panel animate-fade-in">
          <h2>Training Results</h2>
          <p style={{ color: 'var(--text-secondary)', marginBottom: '24px' }}>
            Target: <strong>{targetColumn}</strong> | Task: <strong>{taskType}</strong>
          </p>
          
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '20px' }}>
            {Object.entries(results).map(([modelName, data]) => (
              <div key={modelName} className="glass-panel" style={{ background: 'rgba(255,255,255,0.02)' }}>
                <h3 style={{ textTransform: 'capitalize', marginBottom: '16px', color: 'var(--primary-color)' }}>
                  {modelName.replace('_', ' ')}
                </h3>
                
                <h4 style={{ color: 'var(--text-secondary)', marginBottom: '12px', fontSize: '0.9rem', textTransform: 'uppercase' }}>Metrics</h4>
                <div style={{ marginBottom: '24px' }}>
                  {Object.entries(data.metrics).map(([metricName, value]) => (
                    <div key={metricName} style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                      <span style={{ color: 'var(--text-secondary)', textTransform: 'uppercase', fontSize: '0.85rem' }}>
                        {metricName.replace('_', ' ')}
                      </span>
                      <span style={{ fontWeight: '600' }}>
                        {typeof value === 'number' ? value.toFixed(4) : value}
                      </span>
                    </div>
                  ))}
                </div>

                {data.feature_importances && Object.keys(data.feature_importances).length > 0 && (
                  <>
                    <h4 style={{ color: 'var(--text-secondary)', marginBottom: '12px', fontSize: '0.9rem', textTransform: 'uppercase' }}>Top Features</h4>
                    <div>
                      {Object.entries(data.feature_importances).map(([featureName, importance]) => (
                        <div key={featureName} style={{ marginBottom: '8px' }}>
                          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', marginBottom: '4px' }}>
                            <span style={{ color: 'var(--text-primary)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', maxWidth: '70%' }} title={featureName}>
                              {featureName}
                            </span>
                            <span style={{ color: 'var(--text-secondary)' }}>{importance.toFixed(3)}</span>
                          </div>
                          <div style={{ width: '100%', backgroundColor: 'rgba(255,255,255,0.1)', height: '6px', borderRadius: '3px', overflow: 'hidden' }}>
                            <div style={{ width: `${Math.min(100, importance * 100 * 3)}%`, backgroundColor: 'var(--secondary-color)', height: '100%', borderRadius: '3px' }}></div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </>
                )}
              </div>
            ))}
          </div>
          
          <div style={{ marginTop: '32px', textAlign: 'center' }}>
            <button className="btn btn-outline" onClick={() => { setResults(null); setSchema(null); setFile(null); }}>
              Start Over
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
