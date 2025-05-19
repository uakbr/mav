import React from 'react';
import PanelBase from './PanelBase';
import { ModelMeasurement } from '../../types/mav';

interface TopPredictionsPanelProps {
  measurement: ModelMeasurement;
}

const TopPredictionsPanel: React.FC<TopPredictionsPanelProps> = ({ measurement }) => {
  const { decoded_tokens, top_probs, top_ids, logits } = measurement;
  
  return (
    <PanelBase title="Top Predictions" borderColor="border-blue-500">
      <div className="space-y-1">
        {decoded_tokens.map((token, index) => {
          const prob = top_probs[index];
          const logit = logits[0][logits[0].length - 1][top_ids[index]];
          
          return (
            <div key={index} className="flex">
              <span className="text-magenta-500 font-bold w-24 truncate">{token}</span>
              <span className="text-yellow-500 w-20">{(prob * 100).toFixed(1)}%</span>
              <span className="text-cyan-500">{logit.toFixed(1)}</span>
            </div>
          );
        })}
      </div>
    </PanelBase>
  );
};

export default TopPredictionsPanel; 