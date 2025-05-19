import React from 'react';
import PanelBase from './PanelBase';
import { ModelMeasurement } from '../../types/mav';

interface AttentionEntropyPanelProps {
  measurement: ModelMeasurement;
  maxBarLength?: number;
}

const AttentionEntropyPanel: React.FC<AttentionEntropyPanelProps> = ({ 
  measurement,
  maxBarLength = 35
}) => {
  const { attention_entropy_values, attention_entropy_values_normalized } = measurement;
  
  return (
    <PanelBase title="Attention Entropy" borderColor="border-magenta-500">
      <div className="space-y-1">
        {attention_entropy_values.map((entropyVal, i) => {
          const entropyNorm = attention_entropy_values_normalized[i][0]; // First value
          const barLength = Math.abs(entropyNorm);
          
          return (
            <div key={i} className="flex items-center">
              <span className="text-white font-bold w-16">Layer {i+1}</span>
              <span className="text-yellow-500 mr-2">:</span>
              <div 
                className="h-4 bg-purple-500"
                style={{ width: `${(barLength / maxBarLength) * 100}%` }}
              ></div>
              <span className="ml-2 text-white">{entropyVal[0].toFixed(1)}</span>
            </div>
          );
        })}
      </div>
    </PanelBase>
  );
};

export default AttentionEntropyPanel; 