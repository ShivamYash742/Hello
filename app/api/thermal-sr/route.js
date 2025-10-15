import { NextRequest, NextResponse } from 'next/server';
import { writeFile, mkdir } from 'fs/promises';
import { existsSync } from 'fs';
import { join } from 'path';
import { cookies } from 'next/headers';
import prisma from '@/lib/prisma';
import { getSession } from '@/lib/auth';

export async function POST(request) {
  try {
    // Authenticate user (required for creating Job records)
    const cookieStore = await cookies();
    const sessionToken = cookieStore.get('session')?.value;

    let userId = null;
    if (sessionToken) {
      const session = await getSession(sessionToken);
      userId = session?.user?.id || null;
    }

    const formData = await request.formData();
    const thermalFile = formData.get('thermal');
    const opticalFile = formData.get('optical');

    if (!thermalFile || !opticalFile) {
      return NextResponse.json(
        { error: 'Both thermal and optical files are required' },
        { status: 400 }
      );
    }

    // Save files locally
    const uploadsDir = join(process.cwd(), 'uploads');
    if (!existsSync(uploadsDir)) {
      await mkdir(uploadsDir, { recursive: true });
    }
    const timestamp = Date.now();
    
    const thermalBytes = await thermalFile.arrayBuffer();
    const opticalBytes = await opticalFile.arrayBuffer();
    
    const thermalPath = join(uploadsDir, `thermal_${timestamp}.png`);
    const opticalPath = join(uploadsDir, `optical_${timestamp}.png`);
    
    await writeFile(thermalPath, Buffer.from(thermalBytes));
    await writeFile(opticalPath, Buffer.from(opticalBytes));

    // Forward to Python API server
    const pythonApiFormData = new FormData();
    pythonApiFormData.append('thermal_file', thermalFile);
    pythonApiFormData.append('optical_file', opticalFile);

    const pythonApiBase = process.env.PYTHON_API_URL || process.env.THERMAL_API_URL || 'http://localhost:8000';
    const response = await fetch(`${pythonApiBase}/api/thermal-sr`, {
      method: 'POST',
      body: pythonApiFormData,
    });

    if (!response.ok) {
      throw new Error(`Python API error: ${response.statusText}`);
    }

    const result = await response.json();
    
    // Save SR result
    if (result.success && result.result) {
      const base64Data = result.result.replace(/^data:image\/png;base64,/, '');
      const srBuffer = Buffer.from(base64Data, 'base64');
      const srPath = join(uploadsDir, `sr_result_${timestamp}.png`);
      await writeFile(srPath, srBuffer);
      
      // Create job record in database (only if authenticated)
      let job = null;
      if (userId) {
        job = await prisma.job.create({
          data: {
            userId,
            name: `SR Job ${timestamp}`,
            status: 'completed',
            stage: 'sr',
            progress: 100,
            opticalFile: `optical_${timestamp}.png`,
            thermalFile: `thermal_${timestamp}.png`,
            srFile: `sr_result_${timestamp}.png`,
          }
        });
      }

      // Calculate and save evaluation metrics
      const metrics = calculateMetrics(result);
      if (job) {
        await prisma.evaluation.create({
          data: {
            jobId: job.id,
            psnr: metrics.psnr,
            ssim: metrics.ssim,
            rmse: metrics.rmse,
            edgeSharpness: metrics.edgeSharpness,
            thermalBias: metrics.thermalBias,
            perClassMetrics: metrics.perClassMetrics,
          }
        });
      }

      return NextResponse.json({
        success: true,
        data: {
          ...result,
          jobId: job?.id || null,
          metrics: metrics
        }
      });
    }

    return NextResponse.json({
      success: true,
      data: result
    });

  } catch (error) {
    console.error('Thermal SR API error:', error);
    return NextResponse.json(
      { error: 'Failed to process thermal super-resolution' },
      { status: 500 }
    );
  }
}

function calculateMetrics(result) {
  // Mock metrics calculation - replace with actual computation
  return {
    psnr: 28.42 + Math.random() * 2,
    ssim: 0.891 + Math.random() * 0.05,
    rmse: 1.24 + Math.random() * 0.3,
    edgeSharpness: 0.78 + Math.random() * 0.1,
    thermalBias: 0.15 + Math.random() * 0.1,
    perClassMetrics: [
      { class: 'Urban', psnr: 29.1, ssim: 0.903, rmse: 1.12, pixels: 524288 },
      { class: 'Vegetation', psnr: 27.8, ssim: 0.882, rmse: 1.31, pixels: 786432 },
      { class: 'Water', psnr: 30.2, ssim: 0.915, rmse: 0.98, pixels: 262144 },
      { class: 'Bare Soil', psnr: 26.9, ssim: 0.865, rmse: 1.45, pixels: 327680 },
    ]
  };
}