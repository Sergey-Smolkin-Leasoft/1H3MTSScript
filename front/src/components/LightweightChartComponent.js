
import React, { useEffect, useRef, memo, useCallback } from 'react'; 
import { createChart, LineStyle, TickMarkType, ColorType, CrosshairMode } from 'lightweight-charts';

// Логирование импортированной функции createChart
console.log("[LWC Component] Imported createChart function:", createChart);
console.log("[LWC Component] Type of createChart:", typeof createChart);
if (typeof createChart === 'function') {
    console.log("[LWC Component] createChart.toString().substring(0, 250):", createChart.toString().substring(0, 250));
}


const LightweightChartComponent = memo(({ ohlcData, interval, marketStructureData, theme = 'light' }) => {
  const chartContainerRef = useRef(null);
  const chartRef = useRef(null);
  const candlestickSeriesRef = useRef(null);
  const structureDrawingElementsRef = useRef({ lines: [], priceLines: [] });

  const clearPreviousStructureDrawings = useCallback(() => {
    if (chartRef.current) {
      structureDrawingElementsRef.current.lines.forEach(series => {
        try { chartRef.current.removeSeries(series); } catch(e) { console.warn("Не удалось удалить серию линий структуры:", e); }
      });
      structureDrawingElementsRef.current.priceLines.forEach(priceLine => {
        try { 
          if (candlestickSeriesRef.current) {
            candlestickSeriesRef.current.removePriceLine(priceLine); 
          }
        } catch(e) { console.warn("Не удалось удалить ценовую линию структуры:", e); }
      });
    }
    structureDrawingElementsRef.current = { lines: [], priceLines: [] };
  }, []); 

  const drawMarketStructure = useCallback((structureData) => {
    if (!chartRef.current || !candlestickSeriesRef.current || !structureData) {
      return;
    }
    clearPreviousStructureDrawings(); 

    for (let i = 0; i < structureData.length - 1; i++) {
      const p1 = structureData[i];
      const p2 = structureData[i+1];
      const lineSeries = chartRef.current.addLineSeries({
        color: theme === 'light' ? '#2563eb' : '#60a5fa',
        lineWidth: 2,
        lineStyle: LineStyle.Dashed,
        lastValueVisible: false,
        priceLineVisible: false,
      });
      lineSeries.setData([{ time: p1.time, value: p1.value }, { time: p2.time, value: p2.value }]);
      structureDrawingElementsRef.current.lines.push(lineSeries);

      if (p1.label) {
        const priceLine = candlestickSeriesRef.current.createPriceLine({
          price: p1.value,
          color: theme === 'light' ? '#374151' : '#9ca3af',
          lineWidth: 1,
          lineStyle: LineStyle.Solid,
          axisLabelVisible: true,
          title: p1.label,
        });
        structureDrawingElementsRef.current.priceLines.push(priceLine);
      }
    }
    if (structureData.length > 0) {
        const lastPoint = structureData[structureData.length - 1];
        if (lastPoint.label) {
            const priceLine = candlestickSeriesRef.current.createPriceLine({
            price: lastPoint.value, color: theme === 'light' ? '#374151' : '#9ca3af', lineWidth: 1,
            lineStyle: LineStyle.Solid, axisLabelVisible: true, title: lastPoint.label,
            });
            structureDrawingElementsRef.current.priceLines.push(priceLine);
        }
    }
  }, [theme, clearPreviousStructureDrawings]); 

  const addDaySeparators = useCallback((data) => {
    if (!chartRef.current || !candlestickSeriesRef.current || !data || data.length === 0) return;

    let lastDay = null;
    const priceMin = Math.min(...data.map(d => d.low));
    const priceMax = Math.max(...data.map(d => d.high));

    data.forEach(dataPoint => {
        const date = new Date(dataPoint.time * 1000);
        const day = date.getUTCDate();

        if (lastDay !== null && day !== lastDay && date.getUTCHours() === 0 && date.getUTCMinutes() === 0) {
            const separatorLine = chartRef.current.addLineSeries({
                color: theme === 'light' ? 'rgba(192, 192, 192, 0.7)' : 'rgba(128, 128, 128, 0.5)',
                lineWidth: 1,
                lineStyle: LineStyle.Dashed,
                lastValueVisible: false,
                priceLineVisible: false,
                crosshairMarkerVisible: false,
            });
            separatorLine.setData([
                { time: dataPoint.time, value: priceMin - (priceMax - priceMin) * 0.1 },
                { time: dataPoint.time, value: priceMax + (priceMax - priceMin) * 0.1 }
            ]);
            structureDrawingElementsRef.current.lines.push(separatorLine);
        }
        lastDay = day;
    });
  }, [theme]); 

  useEffect(() => {
    console.log("[LWC Component] useEffect for chart creation/recreation triggered. Interval:", interval, "Theme:", theme);
    if (!chartContainerRef.current) {
        console.warn("[LWC Component] chartContainerRef.current is null in useEffect. Skipping chart creation.");
        return;
    }

    if (chartRef.current) {
      console.log("[LWC Component] Removing previous chart instance.");
      clearPreviousStructureDrawings();
      chartRef.current.remove();
      chartRef.current = null;
      candlestickSeriesRef.current = null;
    }
    
    const chartThemeColors = theme === 'light' ? {
        layout: { background: { type: ColorType.Solid, color: '#FFFFFF' }, textColor: '#333333' },
        grid: { vertLines: { color: '#E0E0E0' }, horzLines: { color: '#E0E0E0' } },
        rightPriceScale: { borderColor: '#C0C0C0' },
        timeScale: { borderColor: '#C0C0C0' },
        candlestick: {
            upColor: '#FFFFFF', downColor: '#000000',
            borderUpColor: '#000000', borderDownColor: '#000000',
            wickUpColor: '#000000', wickDownColor: '#000000',
        }
    } : { 
        layout: { background: { type: ColorType.Solid, color: '#111827' }, textColor: 'rgba(209, 213, 219, 0.9)' },
        grid: { vertLines: { color: 'rgba(75, 85, 99, 0.5)' }, horzLines: { color: 'rgba(75, 85, 99, 0.5)' } },
        rightPriceScale: { borderColor: 'rgba(107, 114, 128, 0.8)'},
        timeScale: { borderColor: 'rgba(107, 114, 128, 0.8)'},
        candlestick: {
            upColor: 'rgba(16, 185, 129, 0.8)', downColor: 'rgba(239, 68, 68, 0.8)',
            borderDownColor: 'rgba(239, 68, 68, 1)', borderUpColor: 'rgba(16, 185, 129, 1)',
            wickDownColor: 'rgba(239, 68, 68, 1)', wickUpColor: 'rgba(16, 185, 129, 1)',
        }
    };
    
    console.log("[LWC Component] Attempting to create chart. Container clientWidth:", chartContainerRef.current.clientWidth);
    const newChart = createChart(chartContainerRef.current, {
      width: chartContainerRef.current.clientWidth,
      height: chartContainerRef.current.clientHeight,
      layout: chartThemeColors.layout,
      grid: chartThemeColors.grid,
      crosshair: { mode: CrosshairMode.Normal },
      rightPriceScale: chartThemeColors.rightPriceScale,
      timeScale: {
        ...chartThemeColors.timeScale,
        timeVisible: true,
        secondsVisible: interval === '5min',
        tickMarkFormatter: (time, tickMarkType, locale) => {
            const d = new Date(time * 1000);
            if (tickMarkType === TickMarkType.Year ||
                tickMarkType === TickMarkType.Month ||
                (tickMarkType === TickMarkType.DayOfMonth && d.getUTCHours() === 0 && d.getUTCMinutes() === 0) ) {
                 return new Intl.DateTimeFormat(locale, { day: 'numeric', month: 'short' }).format(d);
            }
            return ''; 
        },
      },
    });
    chartRef.current = newChart;
    console.log("[LWC Component] Chart object created:", newChart);
    if (newChart && typeof newChart.addCandlestickSeries === 'function') {
        console.log("[LWC Component] newChart.addCandlestickSeries IS a function. Proceeding.");
    } else {
        console.error("[LWC Component] CRITICAL: newChart.addCandlestickSeries is NOT a function. newChart:", newChart);
        if(newChart) {
            console.log("[LWC Component] Properties of newChart object:");
            for(const key in newChart) {
                if (Object.prototype.hasOwnProperty.call(newChart, key)) { // Check if property is own
                     console.log(`  ${key}: ${typeof newChart[key]}`);
                }
            }
             try { // Try to get prototype
                console.log("[LWC Component] Prototype of newChart:", Object.getPrototypeOf(newChart));
             } catch(e) {
                console.warn("[LWC Component] Could not get prototype of newChart:", e);
             }
        }
        return; 
    }


    const newSeries = newChart.addCandlestickSeries(chartThemeColors.candlestick); 
    candlestickSeriesRef.current = newSeries;
    console.log("[LWC Component] Candlestick series added.");


    const adaptChartContainerHeight = () => {
        if (!chartContainerRef.current) return;
        const headerHeight = document.querySelector('header')?.offsetHeight || 0;
        const buttonsHeight = document.querySelector('.button-group')?.offsetHeight || 0;
        const paddingAndMargins = 80; 
        const availableHeight = window.innerHeight - headerHeight - buttonsHeight - paddingAndMargins;
        const newHeight = Math.max(300, availableHeight);
        chartContainerRef.current.style.height = `${newHeight}px`; 
         if (chartRef.current) {
            chartRef.current.applyOptions({width: chartContainerRef.current.clientWidth, height: newHeight});
        }
    };
    
    adaptChartContainerHeight();
    window.addEventListener('resize', adaptChartContainerHeight);

    return () => {
      console.log("[LWC Component] Cleanup: Removing event listener and chart.");
      window.removeEventListener('resize', adaptChartContainerHeight);
      if (chartRef.current) {
        clearPreviousStructureDrawings();
        chartRef.current.remove();
        chartRef.current = null;
      }
    };
  }, [interval, theme, clearPreviousStructureDrawings]); 

  useEffect(() => {
    console.log("[LWC Component] useEffect for ohlcData/marketStructureData update. ohlcData length:", ohlcData ? ohlcData.length : 0, "marketStructureData:", marketStructureData);
    if (candlestickSeriesRef.current && ohlcData && ohlcData.length > 0) {
      console.log("[LWC Component] Setting data to candlestick series.");
      candlestickSeriesRef.current.setData(ohlcData);
      addDaySeparators(ohlcData);
      if (marketStructureData) {
        drawMarketStructure(marketStructureData);
      }
      if (chartRef.current) { 
        console.log("[LWC Component] Fitting content.");
        chartRef.current.timeScale().fitContent();
      }
    } else if (candlestickSeriesRef.current && (!ohlcData || ohlcData.length === 0)) {
        console.log("[LWC Component] No ohlcData, clearing candlestick series.");
        candlestickSeriesRef.current.setData([]); 
        clearPreviousStructureDrawings();
    }
  }, [ohlcData, marketStructureData, theme, addDaySeparators, drawMarketStructure, clearPreviousStructureDrawings]);

  return <div ref={chartContainerRef} className="w-full h-[500px] md:h-[600px] border border-gray-200 bg-white rounded-lg shadow-xl p-1" />;
});

export default LightweightChartComponent;
