import React from 'react';
import PanelBase from './PanelBase';
import { ModelMeasurement } from '../../types/mav';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';

interface OutputDistributionPanelProps {
  measurement: ModelMeasurement;
  numBins?: number;
}

const OutputDistributionPanel: React.FC<OutputDistributionPanelProps> = ({ 
  measurement,
  numBins = 20
}) => {
  const { next_token_probs } = measurement;
  
  // Process data for visualization
  const sortedProbs = [...next_token_probs].sort((a, b) => a - b).slice(-100); // Top 100 probs
  
  // Create bins for the histogram
  const data = [];
  for (let i = 0; i < numBins; i++) {
    const binStart = Math.floor((i * sortedProbs.length) / numBins);
    const binEnd = Math.floor(((i + 1) * sortedProbs.length) / numBins);
    const binProbs = sortedProbs.slice(binStart, binEnd);
    const binSum = binProbs.reduce((sum, val) => sum + val, 0);
    
    data.push({
      bin: i,
      value: binSum,
      label: sortedProbs[binEnd - 1]?.toFixed(4) || '0.0000'
    });
  }
  
  return (
    <PanelBase title="Output Distribution" borderColor="border-yellow-500">
      <ResponsiveContainer width="100%" height={250}>
        <BarChart data={data.reverse()} layout="vertical">
          <XAxis type="number" />
          <YAxis 
            dataKey="label" 
            type="category"
            tick={{ fill: '#FFCC00' }} 
            width={60}
          />
          <Tooltip 
            formatter={(value: number) => [`${value.toFixed(4)}`, 'Probability Sum']}
            labelFormatter={(label) => `Bin ${label}`}
          />
          <Bar dataKey="value" fill="#00AAFF" />
        </BarChart>
      </ResponsiveContainer>
    </PanelBase>
  );
};

export default OutputDistributionPanel; 