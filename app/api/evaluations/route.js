import { NextResponse } from 'next/server';
import { PrismaClient } from '@prisma/client';

const prisma = new PrismaClient();

export async function GET() {
  try {
    const evaluations = await prisma.evaluation.findMany({
      include: {
        job: true
      },
      orderBy: {
        createdAt: 'desc'
      },
      take: 1 // Get latest evaluation
    });

    if (evaluations.length === 0) {
      return NextResponse.json({
        success: false,
        message: 'No evaluations found'
      });
    }

    const latest = evaluations[0];
    
    return NextResponse.json({
      success: true,
      data: {
        metrics: {
          psnr: latest.psnr,
          ssim: latest.ssim,
          rmse: latest.rmse,
          edgeSharpness: latest.edgeSharpness,
          thermalBias: latest.thermalBias
        },
        perClassMetrics: latest.perClassMetrics,
        job: {
          id: latest.job.id,
          name: latest.job.name,
          opticalFile: latest.job.opticalFile,
          thermalFile: latest.job.thermalFile,
          srFile: latest.job.srFile,
          createdAt: latest.job.createdAt
        }
      }
    });
  } catch (error) {
    console.error('Evaluations API error:', error);
    return NextResponse.json(
      { error: 'Failed to fetch evaluations' },
      { status: 500 }
    );
  }
}