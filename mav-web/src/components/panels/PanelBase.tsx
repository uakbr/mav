import React from 'react';

interface PanelBaseProps {
  title: string;
  children: React.ReactNode;
  borderColor?: string;
}

const PanelBase: React.FC<PanelBaseProps> = ({ title, children, borderColor = 'border-blue-500' }) => {
  return (
    <div className={`border ${borderColor} rounded-md p-4 h-full flex flex-col`}>
      <h3 className="text-lg font-mono font-bold mb-2">{title}</h3>
      <div className="font-mono text-sm overflow-auto flex-1">
        {children}
      </div>
    </div>
  );
};

export default PanelBase; 