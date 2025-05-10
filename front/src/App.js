// src/App.js
// Главный компонент приложения

import React, { useState, useCallback, useEffect } from 'react';
import LightweightChartComponent from './components/LightweightChartComponent'; 

// API ключ от Twelve Data.
const API_KEY = "9c614fea46d04e3d8c4f3f76b0541ab6";
const SYMBOL = "EUR/USD";


function App() {
  const [currentInterval, setCurrentInterval] = useState('1h'); 
  const [ohlcData, setOhlcData] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [userMessage, setUserMessage] = useState({ text: '', type: '' }); 

  const [marketStructureData, setMarketStructureData] = useState(null);

  const showUserMessage = (text, type = 'success') => {
    setUserMessage({ text, type });
    setTimeout(() => {
      setUserMessage({ text: '', type: '' });
    }, 5000);
  };
  
  const clearMarketStructure = useCallback(() => {
    setMarketStructureData(null);
  }, []);


  const fetchData = useCallback(async (intervalToFetch) => {
    console.log(`App: fetchData вызвана для интервала ${intervalToFetch}`);
    if (isLoading) { // Предотвращаем повторный вызов, если загрузка уже идет
        console.log("App: fetchData - загрузка уже в процессе, пропуск.");
        return;
    }
    setIsLoading(true);
    setError(null);
    
    if (marketStructureData) { // Очищаем структуру, если она была, перед новым запросом данных
        clearMarketStructure();
    }


    if (!API_KEY || API_KEY === "YOUR_API_KEY") {
      showUserMessage("Пожалуйста, укажите ваш API ключ Twelve Data.", "error");
      setIsLoading(false);
      return;
    }

    const outputSize = intervalToFetch === '1h' ? 300 : 500;
    const apiUrl = `https://api.twelvedata.com/time_series?symbol=${SYMBOL}&interval=${intervalToFetch}&apikey=${API_KEY}&outputsize=${outputSize}`;

    try {
      const response = await fetch(apiUrl);
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(`Ошибка API: ${response.status} - ${errorData.message || 'Не удалось получить данные'}`);
      }
      const data = await response.json();

      if (data.status === "error" || !data.values) {
        throw new Error(`Ошибка от API Twelve Data: ${data.message || 'Данные не получены.'}`);
      }

      const formattedData = data.values.map(item => ({
        time: Date.parse(item.datetime) / 1000,
        open: parseFloat(item.open),
        high: parseFloat(item.high),
        low: parseFloat(item.low),
        close: parseFloat(item.close)
      })).reverse();
      
      setOhlcData(formattedData); // Устанавливаем новые данные
      showUserMessage(`Данные для ${intervalToFetch} успешно загружены.`, 'success');
    } catch (err) {
      console.error("Ошибка при загрузке данных:", err);
      setError(err.message);
      showUserMessage(`Ошибка загрузки данных: ${err.message}`, "error");
      setOhlcData([]); 
    } finally {
      setIsLoading(false);
    }
  }, [clearMarketStructure, isLoading, marketStructureData]); // Убрали marketStructureData из зависимостей, добавили isLoading

  const handleIntervalChange = (newInterval) => {
    if (newInterval !== currentInterval) {
        setCurrentInterval(newInterval);
    }
  };

  useEffect(() => {
    console.log(`App: useEffect[currentInterval] - загрузка для ${currentInterval}`);
    fetchData(currentInterval);
  }, [currentInterval, fetchData]);


  const handleDrawMarketStructure = () => {
    if (ohlcData.length < 20) {
      showUserMessage("Недостаточно данных на графике для отрисовки структуры.", "error");
      return;
    }
    const examplePoints = [
      { time: ohlcData[ohlcData.length - 20].time, value: ohlcData[ohlcData.length - 20].low, label: "LL" },
      { time: ohlcData[ohlcData.length - 15].time, value: ohlcData[ohlcData.length - 15].high, label: "LH" },
      { time: ohlcData[ohlcData.length - 10].time, value: ohlcData[ohlcData.length - 10].low, label: "LL" },
      { time: ohlcData[ohlcData.length - 5].time, value: ohlcData[ohlcData.length - 5].high, label: "LH" }
    ];
    setMarketStructureData(examplePoints);
    showUserMessage("Пример рыночной структуры сгенерирован.", "success");
  };


  return (
    <div className="min-h-screen bg-gray-50 text-gray-800 p-4 md:p-8 font-['Inter',_sans-serif]">
      {userMessage.text && (
        <div 
          className={`fixed top-5 left-1/2 -translate-x-1/2 p-4 rounded-md shadow-lg z-50 text-white ${userMessage.type === 'success' ? 'bg-green-500' : 'bg-red-500'}`}
        >
          {userMessage.text}
        </div>
      )}

      <header className="text-center mb-8">
        <h1 className="text-3xl font-bold text-blue-600">График EUR/USD (React)</h1>
        <p className="text-gray-600">Интерактивный график с Lightweight Charts и React</p>
      </header>

      <div className="flex justify-center items-center space-x-2 mb-6">
        <button 
          onClick={() => handleIntervalChange('5min')}
          className={`px-4 py-2 rounded-md shadow-sm border border-gray-300
                      ${currentInterval === '5min' ? 'bg-blue-400 text-white font-bold' : 'bg-white text-gray-700 hover:bg-gray-200'}`}
        >
          5 минут
        </button>
        <button 
          onClick={() => handleIntervalChange('1h')}
          className={`px-4 py-2 rounded-md shadow-sm border border-gray-300
                      ${currentInterval === '1h' ? 'bg-blue-400 text-white font-bold' : 'bg-white text-gray-700 hover:bg-gray-200'}`}
        >
          1 час
        </button>
        <button 
          onClick={handleDrawMarketStructure}
          className="px-4 py-2 bg-white text-gray-700 rounded-md shadow-sm border border-gray-300 hover:bg-gray-200"
        >
          Нарисовать структуру (Пример)
        </button>
         <button 
          onClick={clearMarketStructure}
          className="px-4 py-2 bg-white text-gray-700 rounded-md shadow-sm border border-gray-300 hover:bg-gray-200"
        >
          Очистить структуру
        </button>
      </div>

      {isLoading && <p className="text-center text-blue-500">Загрузка данных...</p>}
      {error && <p className="text-center text-red-500">Ошибка: {error}</p>}
      
      <div className="max-w-4xl mx-auto">
        <LightweightChartComponent 
          ohlcData={ohlcData} 
          interval={currentInterval}
          marketStructureData={marketStructureData} 
          theme="light" 
        />
      </div>
    </div>
  );
}

export default App;
