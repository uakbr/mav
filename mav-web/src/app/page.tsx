"use client";

import React, { useState, useRef, useEffect } from 'react';
import MAVContainer from '../components/MAVContainer';
import { ModelMeasurement, MAVParams, ErrorResponse, DoneResponse } from '../types/mav';

export default function Home() {
  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [measurement, setMeasurement] = useState<ModelMeasurement | null>(null);
  const [params, setParams] = useState<MAVParams>({
    model: 'gpt2',
    prompt: 'Hello world',
    max_new_tokens: 50,
    temperature: 0.0,
    top_k: 50,
    top_p: 1.0,
    min_p: 0.0,
    repetition_penalty: 1.0,
    refresh_rate: 0.1,
    aggregation: 'l2',
    scale: 'linear',
    max_bar_length: 35
  });
  
  const eventSourceRef = useRef<EventSource | null>(null);
  
  useEffect(() => {
    return () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }
    };
  }, []);
  
  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement>) => {
    const { name, value, type } = e.target as HTMLInputElement;
    
    // Convert numeric values from strings to numbers
    const parsedValue = ['temperature', 'top_k', 'top_p', 'min_p', 'repetition_penalty', 'refresh_rate', 'max_new_tokens', 'max_bar_length'].includes(name)
      ? (type === 'number' ? parseFloat(value) : value)
      : value;
      
    setParams((prev: MAVParams) => ({ ...prev, [name]: parsedValue }));
  };
  
  const startMAV = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (isRunning) {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
        eventSourceRef.current = null;
      }
      setIsRunning(false);
      return;
    }
    
    setError(null);
    setIsRunning(true);
    
    // Create EventSource for streaming
    const eventSource = new EventSource(`/api/run?params=${encodeURIComponent(JSON.stringify(params))}`);
    eventSourceRef.current = eventSource;
    
    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        
        // Check if it's an error message
        if ('error' in data) {
          const errorData = data as ErrorResponse;
          setError(errorData.error);
          setIsRunning(false);
          eventSource.close();
          return;
        }
        
        // Check if it's a 'done' message
        if ('done' in data) {
          const doneData = data as DoneResponse;
          if (doneData.done) {
            setIsRunning(false);
            eventSource.close();
            return;
          }
        }
        
        // Otherwise, it's a measurement
        setMeasurement(data as ModelMeasurement);
      } catch (err) {
        console.error('Failed to parse event data:', err);
        setError('Failed to parse data from server');
        setIsRunning(false);
        eventSource.close();
      }
    };
    
    eventSource.onerror = () => {
      setError('Connection to server failed');
      setIsRunning(false);
      eventSource.close();
      eventSourceRef.current = null;
    };
  };
  
  return (
    <main className="min-h-screen p-8">
      <div className="max-w-7xl mx-auto">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold mb-2">MAV - Model Activity Visualizer</h1>
          <p className="text-gray-400">Visualize the internal workings of LLMs as they generate text</p>
        </div>
        
        {error && (
          <div className="mb-6 p-4 bg-red-900/50 border border-red-700 rounded-md text-red-300">
            {error}
          </div>
        )}
        
        <form onSubmit={startMAV} className="mb-8 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          <div className="space-y-2 col-span-1 md:col-span-2 lg:col-span-3">
            <label htmlFor="prompt" className="block text-sm font-medium">Prompt</label>
            <textarea
              id="prompt"
              name="prompt"
              value={params.prompt}
              onChange={handleInputChange}
              className="w-full p-2 bg-gray-800 border border-gray-700 rounded-md text-white"
              rows={3}
              required
            />
          </div>
          
          <div className="space-y-2">
            <label htmlFor="model" className="block text-sm font-medium">Model</label>
            <select
              id="model"
              name="model"
              value={params.model}
              onChange={handleInputChange}
              className="w-full p-2 bg-gray-800 border border-gray-700 rounded-md text-white"
            >
              <option value="gpt2">GPT-2 (Small)</option>
              <option value="gpt2-medium">GPT-2 (Medium)</option>
              <option value="HuggingFaceTB/SmolLM-135M">SmolLM-135M</option>
            </select>
          </div>
          
          <div className="space-y-2">
            <label htmlFor="max_new_tokens" className="block text-sm font-medium">Max Tokens</label>
            <input
              type="number"
              id="max_new_tokens"
              name="max_new_tokens"
              value={params.max_new_tokens}
              onChange={handleInputChange}
              min="1"
              max="200"
              className="w-full p-2 bg-gray-800 border border-gray-700 rounded-md text-white"
            />
          </div>
          
          <div className="space-y-2">
            <label htmlFor="temperature" className="block text-sm font-medium">Temperature</label>
            <input
              type="number"
              id="temperature"
              name="temperature"
              value={params.temperature}
              onChange={handleInputChange}
              min="0"
              max="2"
              step="0.1"
              className="w-full p-2 bg-gray-800 border border-gray-700 rounded-md text-white"
            />
          </div>
          
          <div className="space-y-2">
            <label htmlFor="top_k" className="block text-sm font-medium">Top-K</label>
            <input
              type="number"
              id="top_k"
              name="top_k"
              value={params.top_k}
              onChange={handleInputChange}
              min="0"
              max="100"
              className="w-full p-2 bg-gray-800 border border-gray-700 rounded-md text-white"
            />
          </div>
          
          <div className="space-y-2">
            <label htmlFor="top_p" className="block text-sm font-medium">Top-P</label>
            <input
              type="number"
              id="top_p"
              name="top_p"
              value={params.top_p}
              onChange={handleInputChange}
              min="0"
              max="1"
              step="0.05"
              className="w-full p-2 bg-gray-800 border border-gray-700 rounded-md text-white"
            />
          </div>
          
          <button
            type="submit"
            className={`col-span-1 md:col-span-2 lg:col-span-3 p-3 rounded-md font-medium ${
              isRunning ? 'bg-red-600 hover:bg-red-700' : 'bg-blue-600 hover:bg-blue-700'
            }`}
          >
            {isRunning ? 'Stop' : 'Start Visualization'}
          </button>
        </form>
        
        <MAVContainer 
          measurement={measurement}
          numGridRows={2}
          limitChars={250}
          maxBarLength={35}
        />
      </div>
    </main>
  );
} 