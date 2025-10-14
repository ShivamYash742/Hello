import { NextResponse } from 'next/server';
import { cookies } from 'next/headers';
import { getSession } from '@/lib/auth';
import prisma from '@/lib/prisma';
import { writeFile, mkdir } from 'fs/promises';
import { join } from 'path';
import { existsSync } from 'fs';

export async function POST(request) {
  try {
    // Check authentication
    const cookieStore = await cookies();
    const sessionToken = cookieStore.get('session')?.value;
    
    if (!sessionToken) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    const session = await getSession(sessionToken);
    if (!session) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    const formData = await request.formData();
    const opticalFile = formData.get('optical');
    const thermalFile = formData.get('thermal');

    if (!opticalFile || !thermalFile) {
      return NextResponse.json(
        { error: 'Both optical and thermal files are required' },
        { status: 400 }
      );
    }

    // Create uploads directory if it doesn't exist
    const uploadsDir = join(process.cwd(), 'uploads');
    if (!existsSync(uploadsDir)) {
      await mkdir(uploadsDir, { recursive: true });
    }

    // Save files
    const opticalBytes = await opticalFile.arrayBuffer();
    const thermalBytes = await thermalFile.arrayBuffer();
    
    const opticalBuffer = Buffer.from(opticalBytes);
    const thermalBuffer = Buffer.from(thermalBytes);

    const timestamp = Date.now();
    const opticalPath = join(uploadsDir, `optical_${timestamp}_${opticalFile.name}`);
    const thermalPath = join(uploadsDir, `thermal_${timestamp}_${thermalFile.name}`);

    await writeFile(opticalPath, opticalBuffer);
    await writeFile(thermalPath, thermalBuffer);

    // Create job in database
    const job = await prisma.job.create({
      data: {
        userId: session.user.id,
        name: `Job ${new Date().toLocaleString()}`,
        status: 'queued',
        stage: 'ingest',
        progress: 0,
        opticalFile: opticalPath,
        thermalFile: thermalPath,
        config: {
          opticalName: opticalFile.name,
          thermalName: thermalFile.name,
        },
      },
    });

    return NextResponse.json({
      success: true,
      jobId: job.id,
      message: 'Files uploaded successfully',
    });
  } catch (error) {
    console.error('Upload error:', error);
    return NextResponse.json(
      { error: 'Failed to upload files' },
      { status: 500 }
    );
  }
}

export const config = {
  api: {
    bodyParser: false,
  },
};