import React from 'react';
import PanelBase from './PanelBase';
import { ModelMeasurement } from '../../types/mav';

interface GeneratedTextPanelProps {
  measurement: ModelMeasurement;
  limitChars?: number;
}

const GeneratedTextPanel: React.FC<GeneratedTextPanelProps> = ({ 
  measurement, 
  limitChars = 250 
}) => {
  const { generated_text, predicted_char } = measurement;
  
  // Limit the displayed text to the last N characters
  const displayText = generated_text.slice(-limitChars);
  
  return (
    <PanelBase title="Generated Text" borderColor="border-green-500">
      <div className="whitespace-pre-wrap break-words">
        <span className="text-red-500 font-bold">{displayText}</span>
        <span className="bg-green-500 text-black font-bold">{predicted_char}</span>
      </div>
    </PanelBase>
  );
};

export default GeneratedTextPanel; 