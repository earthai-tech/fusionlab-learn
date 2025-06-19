import React, { useState, useRef, useEffect } from 'react';

// --- Helper Components for UI Structure (similar to shadcn/ui) ---

const Card = ({ children, className = '' }) => (
  <div className={`bg-slate-800 border border-slate-700 rounded-lg shadow-lg ${className}`}>
    {children}
  </div>
);

const CardHeader = ({ children, className = '' }) => <div className={`p-6 ${className}`}>{children}</div>;
const CardTitle = ({ children, className = '' }) => <h3 className={`text-xl font-semibold tracking-tight text-white ${className}`}>{children}</h3>;
const CardDescription = ({ children, className = '' }) => <p className={`text-sm text-slate-400 ${className}`}>{children}</p>;
const CardContent = ({ children, className = '' }) => <div className={`p-6 pt-0 ${className}`}>{children}</div>;
const CardFooter = ({ children, className = '' }) => <div className={`p-6 pt-0 flex items-center ${className}`}>{children}</div>;

const Label = ({ children, htmlFor }) => <label htmlFor={htmlFor} className="text-sm font-medium leading-none text-slate-300 peer-disabled:cursor-not-allowed peer-disabled:opacity-70">{children}</label>;
const Input = (props) => <input {...props} className={`flex h-10 w-full rounded-md border border-slate-600 bg-slate-900 px-3 py-2 text-sm text-white placeholder:text-slate-500 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 focus-visible:ring-offset-2 focus-visible:ring-offset-slate-900 disabled:cursor-not-allowed disabled:opacity-50 ${props.className || ''}`} />;
const Select = ({ children, ...props }) => <select {...props} className={`flex h-10 w-full items-center justify-between rounded-md border border-slate-600 bg-slate-900 px-3 py-2 text-sm text-white ring-offset-slate-900 placeholder:text-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50 [&>span]:line-clamp-1 ${props.className || ''}`}>{children}</select>;
const Button = ({ children, className = '', ...props }) => <button {...props} className={`inline-flex items-center justify-center rounded-md text-sm font-medium ring-offset-slate-900 transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 bg-blue-600 text-white hover:bg-blue-700 h-10 px-4 py-2 ${className}`}>{children}</button>;

// --- SVG Icons ---
const UploadIcon = (props) => (
  <svg {...props} xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
    <polyline points="17 8 12 3 7 8" />
    <line x1="12" x2="12" y1="3" y2="15" />
  </svg>
);
const SettingsIcon = (props) => (
    <svg {...props} xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 0 2l-.15.08a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.38a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1 0-2l.15-.08a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z" />
        <circle cx="12" cy="12" r="3" />
    </svg>
);
const FileTextIcon = (props) => (
    <svg {...props} xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z" />
        <polyline points="14 2 14 8 20 8" />
        <line x1="16" x2="8" y1="13" y2="13" />
        <line x1="16" x2="8" y1="17" y2="17" />
        <line x1="10" x2="8" y1="9" y2="9" />
    </svg>
);

// --- Main App Component ---
const SubsidenceForecasterApp = () => {
    // --- State Management ---
    const [file, setFile] = useState(null);
    const [fileName, setFileName] = useState('');
    const [config, setConfig] = useState({
        modelName: 'TransFlowSubsNet',
        trainEndYear: 2022,
        forecastStartYear: 2023,
        forecastHorizonYears: 3,
        timeSteps: 5,
        pdeMode: 'both',
        coeffC: 'learnable',
        lambdaCons: 1.0,
        lambdaGw: 1.0,
        quantiles: '0.1, 0.5, 0.9',
        epochs: 50,
        learningRate: 0.001,
        batchSize: 256,
    });
    
    const [isProcessing, setIsProcessing] = useState(false);
    const [logs, setLogs] = useState([]);
    const [statusMessage, setStatusMessage] = useState('Ready. Configure parameters and upload a file to start.');
    const [resultsDataframe, setResultsDataframe] = useState(null);
    const [resultPlots, setResultPlots] = useState([]);
    
    const logContainerRef = useRef(null);

    // Auto-scroll log container
    useEffect(() => {
        if (logContainerRef.current) {
            logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight;
        }
    }, [logs]);

    // --- Handlers ---
    const handleFileChange = (e) => {
        const selectedFile = e.target.files[0];
        if (selectedFile) {
            setFile(selectedFile);
            setFileName(selectedFile.name);
            addLog(`File selected: "${selectedFile.name}"`);
        }
    };

    const handleConfigChange = (e) => {
        const { name, value } = e.target;
        setConfig(prev => ({ ...prev, [name]: value }));
    };
    
    const addLog = (message) => {
        const timestamp = new Date().toLocaleTimeString();
        setLogs(prev => [...prev, `[${timestamp}] ${message}`]);
    };

    const handleRunForecast = async () => {
        if (!file) {
            addLog("Error: No data file uploaded. Please select a file.");
            return;
        }

        setIsProcessing(true);
        setResultsDataframe(null);
        setResultPlots([]);
        addLog("--- Starting Forecasting Workflow ---");
        
        const steps = [
            { msg: "Step 1: Validating configuration and data...", delay: 500 },
            { msg: "Step 2: Loading and preprocessing data...", delay: 1500 },
            { msg: "Step 3: Generating PINN data sequences...", delay: 2000 },
            { msg: "Step 4: Creating TensorFlow datasets...", delay: 1000 },
            { msg: `Step 5: Training ${config.modelName} model for ${config.epochs} epochs...`, delay: 5000 },
            { msg: "Step 6: Generating predictions on test data...", delay: 1500 },
            { msg: "Step 7: Formatting results and generating visualizations...", delay: 2000 },
        ];

        for (const step of steps) {
            setStatusMessage(step.msg);
            addLog(step.msg);
            await new Promise(resolve => setTimeout(resolve, step.delay));
        }

        addLog("--- Workflow Complete ---");
        setStatusMessage("Workflow finished successfully.");

        // Simulate final results
        const dummyCsvData = `sample_idx,forecast_step,coord_t,coord_x,coord_y,subsidence_q10,subsidence_q50,subsidence_q90,subsidence_actual\n0,1,2023.0,113.5,22.5,-10.5,-10.1,-9.8,-10.2\n0,2,2024.0,113.5,22.5,-11.2,-10.8,-10.5,-11.0\n1,1,2023.0,113.6,22.6,-12.1,-11.8,-11.5,-11.9`;
        setResultsDataframe(dummyCsvData);
        setResultPlots([
            `https://placehold.co/600x400/1e293b/ffffff?text=Training+History+Plot`,
            `https://placehold.co/600x400/1e293b/ffffff?text=Forecast+Visualization`
        ]);
        
        setIsProcessing(false);
    };
    
    const handleReset = () => {
        setFile(null);
        setFileName('');
        setIsProcessing(false);
        setLogs([]);
        setStatusMessage('Ready. Configure parameters and upload a file to start.');
        setResultsDataframe(null);
        setResultPlots([]);
    }

    // --- Render ---
    return (
        <div className="bg-slate-900 min-h-screen font-sans text-slate-300 p-4 sm:p-6 lg:p-8">
            <div className="max-w-7xl mx-auto">
                <header className="mb-8 text-center">
                    <h1 className="text-4xl font-bold text-white tracking-tight">Subsidence Forecasting Tool</h1>
                    <p className="mt-2 text-lg text-slate-400">A GUI for the `fusionlab-learn` PINN workflow.</p>
                </header>

                <div className="flex flex-col lg:flex-row gap-8">
                    {/* --- Configuration Panel (Left) --- */}
                    <div className="lg:w-1/3 space-y-6">
                        <Card>
                            <CardHeader>
                                <CardTitle className="flex items-center gap-2"><UploadIcon className="w-5 h-5"/> Data Input</CardTitle>
                                <CardDescription>Upload your dataset in CSV format.</CardDescription>
                            </CardHeader>
                            <CardContent>
                                <Input id="file-upload" type="file" accept=".csv" onChange={handleFileChange} className="text-sm" />
                                {fileName && <p className="text-xs text-green-400 mt-2">File loaded: {fileName}</p>}
                            </CardContent>
                        </Card>

                        <Card>
                            <CardHeader>
                                <CardTitle className="flex items-center gap-2"><SettingsIcon className="w-5 h-5"/> Workflow Configuration</CardTitle>
                                <CardDescription>Adjust model and training parameters.</CardDescription>
                            </CardHeader>
                            <CardContent className="space-y-4">
                                <div>
                                    <Label htmlFor="modelName">Model</Label>
                                    <Select id="modelName" name="modelName" value={config.modelName} onChange={handleConfigChange}>
                                        <option>TransFlowSubsNet</option>
                                        <option>PIHALNet</option>
                                    </Select>
                                </div>
                                <div className="grid grid-cols-2 gap-4">
                                    <div>
                                        <Label htmlFor="trainEndYear">Train End Year</Label>
                                        <Input id="trainEndYear" name="trainEndYear" type="number" value={config.trainEndYear} onChange={handleConfigChange} />
                                    </div>
                                    <div>
                                        <Label htmlFor="timeSteps">Time Steps (Lookback)</Label>
                                        <Input id="timeSteps" name="timeSteps" type="number" value={config.timeSteps} onChange={handleConfigChange} />
                                    </div>
                                    <div>
                                        <Label htmlFor="forecastStartYear">Forecast Start Year</Label>
                                        <Input id="forecastStartYear" name="forecastStartYear" type="number" value={config.forecastStartYear} onChange={handleConfigChange} />
                                    </div>
                                    <div>
                                        <Label htmlFor="forecastHorizonYears">Forecast Horizon</Label>
                                        <Input id="forecastHorizonYears" name="forecastHorizonYears" type="number" value={config.forecastHorizonYears} onChange={handleConfigChange} />
                                    </div>
                                </div>
                                 <div>
                                    <Label htmlFor="epochs">Training Epochs</Label>
                                    <Input id="epochs" name="epochs" type="number" value={config.epochs} onChange={handleConfigChange} />
                                </div>
                            </CardContent>
                            <CardFooter className="flex-col items-stretch space-y-2">
                                <Button onClick={handleRunForecast} disabled={isProcessing}>
                                    {isProcessing ? (
                                        <>
                                            <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                                                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                                            </svg>
                                            Processing...
                                        </>
                                    ) : "Run Forecast"}
                                </Button>
                                 <Button onClick={handleReset} variant="outline" className="bg-transparent border border-slate-600 hover:bg-slate-700">Reset</Button>
                            </CardFooter>
                        </Card>
                    </div>

                    {/* --- Output Panel (Right) --- */}
                    <div className="lg:w-2/3">
                        <Card className="h-full flex flex-col">
                            <CardHeader>
                                <CardTitle className="flex items-center gap-2"><FileTextIcon className="w-5 h-5" /> Processing Log & Results</CardTitle>
                                <CardDescription>{statusMessage}</CardDescription>
                            </CardHeader>
                            <CardContent className="flex-grow flex flex-col space-y-4 overflow-hidden">
                                <div ref={logContainerRef} className="bg-black/50 p-4 rounded-md h-64 overflow-y-auto font-mono text-xs border border-slate-700">
                                    {logs.map((log, index) => (
                                        <p key={index} className="whitespace-pre-wrap">{log}</p>
                                    ))}
                                </div>

                                {!isProcessing && resultsDataframe && (
                                    <div className="space-y-2">
                                        <h4 className="text-lg font-medium text-white">Results DataFrame</h4>
                                        <div className="bg-black/50 p-4 rounded-md max-h-48 overflow-auto border border-slate-700">
                                            <pre className="text-xs">{resultsDataframe}</pre>
                                        </div>
                                    </div>
                                )}
                                
                                {!isProcessing && resultPlots.length > 0 && (
                                    <div className="space-y-4">
                                        <h4 className="text-lg font-medium text-white">Visualizations</h4>
                                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                            {resultPlots.map((src, index) => (
                                                <img key={index} src={src} alt={`Result plot ${index + 1}`} className="rounded-md border border-slate-700" />
                                            ))}
                                        </div>
                                    </div>
                                )}
                            </CardContent>
                        </Card>
                    </div>
                </div>
            </div>
        </div>
    );
};

// Main App component to export
export default SubsidenceForecasterApp;
