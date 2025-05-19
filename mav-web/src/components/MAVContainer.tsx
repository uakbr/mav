import React from 'react';
import { ModelMeasurement } from '../types/mav';
import GeneratedTextPanel from './panels/GeneratedTextPanel';
import TopPredictionsPanel from './panels/TopPredictionsPanel';
import MlpActivationsPanel from './panels/MlpActivationsPanel';
import AttentionEntropyPanel from './panels/AttentionEntropyPanel';
import OutputDistributionPanel from './panels/OutputDistributionPanel';

interface MAVContainerProps {
  measurement: ModelMeasurement | null;
  numGridRows?: number;
  limitChars?: number;
  maxBarLength?: number;
}

const MAVContainer: React.FC<MAVContainerProps> = ({
  measurement,
  numGridRows = 2,
  limitChars = 250,
  maxBarLength = 35,
}) => {
  if (!measurement) {
    return (
      <div className="w-full h-[500px] flex items-center justify-center">
        <p className="text-xl text-gray-400">Waiting for data...</p>
      </div>
    );
  }

  // Default panels
  const panels = [
    <GeneratedTextPanel
      key="generated-text"
      measurement={measurement}
      limitChars={limitChars}
    />,
    <TopPredictionsPanel
      key="top-predictions"
      measurement={measurement}
    />,
    <OutputDistributionPanel
      key="output-distribution"
      measurement={measurement}
    />,
    <MlpActivationsPanel
      key="mlp-activations"
      measurement={measurement}
      maxBarLength={maxBarLength}
    />,
    <AttentionEntropyPanel
      key="attention-entropy"
      measurement={measurement}
      maxBarLength={maxBarLength}
    />,
  ];

  // Determine grid layout
  const gridTemplateRows = `repeat(${numGridRows}, minmax(0, 1fr))`;
  const columnsPerRow = Math.ceil(panels.length / numGridRows);
  const gridTemplateColumns = `repeat(${columnsPerRow}, minmax(0, 1fr))`;

  return (
    <div 
      className="grid gap-4 h-[800px] w-full"
      style={{ gridTemplateRows, gridTemplateColumns }}
    >
      {panels}
    </div>
  );
};

export default MAVContainer; 