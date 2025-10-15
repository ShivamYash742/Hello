import { NextResponse } from 'next/server';

export async function GET() {
  try {
    // Check if Python API is running
    const response = await fetch('http://localhost:8000/api/health', {
      method: 'GET',
    });

    if (response.ok) {
      const data = await response.json();
      return NextResponse.json({
        status: 'healthy',
        nextjs: 'running',
        python_api: data.status,
        model_loaded: data.model_loaded,
        device: data.device
      });
    } else {
      return NextResponse.json({
        status: 'partial',
        nextjs: 'running',
        python_api: 'unavailable',
        error: 'Python API server not responding'
      }, { status: 503 });
    }
  } catch (error) {
    return NextResponse.json({
      status: 'partial',
      nextjs: 'running',
      python_api: 'unavailable',
      error: error.message
    }, { status: 503 });
  }
}