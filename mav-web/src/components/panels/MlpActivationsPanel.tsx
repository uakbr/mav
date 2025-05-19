import React from 'react';
import PanelBase from './PanelBase';
import { ModelMeasurement } from '../../types/mav';

interface MlpActivationsPanelProps {
  measurement: ModelMeasurement;
  maxBarLength?: number;
}

const MlpActivationsPanel: React.FC<MlpActivationsPanelProps> = ({ 
  measurement,
  maxBarLength = 35
}) => {
  const { mlp_normalized, mlp_activations } = measurement;
  
  return (
    <PanelBase title="MLP Activations" borderColor="border-cyan-500">
      <div className="space-y-1">
        {mlp_normalized.map((mlpNorm, i) => {
          const rawMlp = mlp_activations[i][0]; // Assuming the first value is what we want to show
          const mlpBarLength = Math.abs(mlpNorm[0]); // First value of normalized array
          const isPositive = rawMlp >= 0;
          
          return (
            <div key={i} className="flex items-center">
              <span className="text-white font-bold w-16">Layer {i}</span>
              <span className="text-yellow-500 mr-2">:</span>
              <div 
                className={`h-4 ${isPositive ? 'bg-yellow-500' : 'bg-purple-500'}`} 
                style={{ width: `${(mlpBarLength / maxBarLength) * 100}%` }}
              ></div>
              <span className="ml-2 text-yellow-500">{rawMlp.toFixed(1)}</span>
            </div>
          );
        })}
      </div>
    </PanelBase>
  );
};

export default MlpActivationsPanel; 