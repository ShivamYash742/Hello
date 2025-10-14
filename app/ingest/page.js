'use client';

import { useState, useCallback } from 'react';
import { DashboardLayout } from '@/components/dashboard-layout';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Progress } from '@/components/ui/progress';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Upload, Image as ImageIcon, FileCheck, AlertCircle, ArrowRight } from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import { toast } from 'sonner';
import Link from 'next/link';

export default function IngestPage() {
  const [opticalFile, setOpticalFile] = useState(null);
  const [thermalFile, setThermalFile] = useState(null);
  const [opticalMeta, setOpticalMeta] = useState(null);
  const [thermalMeta, setThermalMeta] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);

  const handleOpticalFile = (e) => {
    const file = e.target.files?.[0];
    if (file) {
      setOpticalFile(file);
      // Mock metadata extraction
      setOpticalMeta({
        name: file.name,
        size: (file.size / 1024 / 1024).toFixed(2) + ' MB',
        type: file.type || 'GeoTIFF',
        resolution: '10m',
        crs: 'EPSG:4326',
        bands: 3,
        width: 2048,
        height: 2048,
      });
    }
  };

  const handleThermalFile = (e) => {
    const file = e.target.files?.[0];
    if (file) {
      setThermalFile(file);
      // Mock metadata extraction
      setThermalMeta({
        name: file.name,
        size: (file.size / 1024 / 1024).toFixed(2) + ' MB',
        type: file.type || 'GeoTIFF',
        resolution: '100m',
        crs: 'EPSG:4326',
        bands: 1,
        width: 512,
        height: 512,
      });
    }
  };

  const handleUpload = async () => {
    if (!opticalFile || !thermalFile) {
      toast.error('Please select both optical and thermal files');
      return;
    }

    setUploading(true);
    setUploadProgress(0);

    // Simulate upload progress
    const interval = setInterval(() => {
      setUploadProgress(prev => {
        if (prev >= 100) {
          clearInterval(interval);
          return 100;
        }
        return prev + 10;
      });
    }, 200);

    try {
      const formData = new FormData();
      formData.append('optical', opticalFile);
      formData.append('thermal', thermalFile);

      const response = await fetch('/api/upload', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Upload failed');
      }

      const data = await response.json();
      toast.success('Files uploaded successfully!');
      
      // Store job ID in session storage for next steps
      if (data.jobId) {
        sessionStorage.setItem('currentJobId', data.jobId);
      }
    } catch (error) {
      console.error('Upload error:', error);
      toast.error('Failed to upload files');
    } finally {
      clearInterval(interval);
      setUploading(false);
      setUploadProgress(100);
    }
  };

  const MetadataCard = ({ title, metadata, icon: Icon }) => (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-lg">
          <Icon className="h-5 w-5 text-[#0E88D3]" />
          {title}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-2">
        {metadata ? (
          <>
            <div className="grid grid-cols-2 gap-2 text-sm">
              <div className="text-muted-foreground">Name:</div>
              <div className="font-medium truncate">{metadata.name}</div>
              
              <div className="text-muted-foreground">Size:</div>
              <div className="font-medium">{metadata.size}</div>
              
              <div className="text-muted-foreground">Resolution:</div>
              <div className="font-medium">{metadata.resolution}</div>
              
              <div className="text-muted-foreground">CRS:</div>
              <div className="font-medium">{metadata.crs}</div>
              
              <div className="text-muted-foreground">Bands:</div>
              <div className="font-medium">{metadata.bands}</div>
              
              <div className="text-muted-foreground">Dimensions:</div>
              <div className="font-medium">{metadata.width} × {metadata.height}</div>
            </div>
            <Badge variant="outline" className="mt-2">
              <FileCheck className="mr-1 h-3 w-3" />
              Validated
            </Badge>
          </>
        ) : (
          <p className="text-sm text-muted-foreground">No file selected</p>
        )}
      </CardContent>
    </Card>
  );

  return (
    <DashboardLayout>
      <div className="container mx-auto px-4 py-8">
        <div className="max-w-6xl mx-auto space-y-8">
          {/* Header */}
          <div>
            <h1 className="text-3xl font-bold mb-2">Data Ingestion</h1>
            <p className="text-muted-foreground">
              Upload high-resolution optical (RGB GeoTIFF) and low-resolution thermal IR imagery
            </p>
          </div>

          {/* CRS Warning */}
          {opticalMeta && thermalMeta && opticalMeta.crs !== thermalMeta.crs && (
            <Alert>
              <AlertCircle className="h-4 w-4" />
              <AlertDescription>
                CRS mismatch detected! Optical ({opticalMeta.crs}) and Thermal ({thermalMeta.crs}) use different coordinate systems.
                Reprojection will be applied during alignment.
              </AlertDescription>
            </Alert>
          )}

          <div className="grid md:grid-cols-2 gap-6">
            {/* Optical File Upload */}
            <Card>
              <CardHeader>
                <CardTitle>HR Optical (RGB)</CardTitle>
                <CardDescription>High-resolution RGB GeoTIFF imagery</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="border-2 border-dashed rounded-lg p-8 text-center hover:border-[#0E88D3] transition-colors">
                  <Upload className="h-12 w-12 mx-auto mb-4 text-muted-foreground" />
                  <Label htmlFor="optical-file" className="cursor-pointer">
                    <span className="text-sm text-[#0E88D3] hover:underline">
                      Choose file or drag and drop
                    </span>
                  </Label>
                  <Input
                    id="optical-file"
                    type="file"
                    accept=".tif,.tiff,.geotiff,image/*"
                    onChange={handleOpticalFile}
                    className="hidden"
                  />
                  <p className="text-xs text-muted-foreground mt-2">
                    GeoTIFF, TIFF (max 500MB)
                  </p>
                </div>
                {opticalFile && (
                  <div className="flex items-center justify-between p-3 bg-muted rounded-md">
                    <span className="text-sm truncate flex-1">{opticalFile.name}</span>
                    <FileCheck className="h-5 w-5 text-green-500 ml-2" />
                  </div>
                )}
              </CardContent>
            </Card>

            {/* Thermal File Upload */}
            <Card>
              <CardHeader>
                <CardTitle>LR Thermal (IR)</CardTitle>
                <CardDescription>Low-resolution thermal infrared GeoTIFF</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="border-2 border-dashed rounded-lg p-8 text-center hover:border-[#F47216] transition-colors">
                  <Upload className="h-12 w-12 mx-auto mb-4 text-muted-foreground" />
                  <Label htmlFor="thermal-file" className="cursor-pointer">
                    <span className="text-sm text-[#F47216] hover:underline">
                      Choose file or drag and drop
                    </span>
                  </Label>
                  <Input
                    id="thermal-file"
                    type="file"
                    accept=".tif,.tiff,.geotiff,image/*"
                    onChange={handleThermalFile}
                    className="hidden"
                  />
                  <p className="text-xs text-muted-foreground mt-2">
                    GeoTIFF, TIFF (max 500MB)
                  </p>
                </div>
                {thermalFile && (
                  <div className="flex items-center justify-between p-3 bg-muted rounded-md">
                    <span className="text-sm truncate flex-1">{thermalFile.name}</span>
                    <FileCheck className="h-5 w-5 text-green-500 ml-2" />
                  </div>
                )}
              </CardContent>
            </Card>
          </div>

          {/* Metadata Display */}
          <div className="grid md:grid-cols-2 gap-6">
            <MetadataCard
              title="Optical Metadata"
              metadata={opticalMeta}
              icon={ImageIcon}
            />
            <MetadataCard
              title="Thermal Metadata"
              metadata={thermalMeta}
              icon={ImageIcon}
            />
          </div>

          {/* Upload Progress */}
          {uploading && (
            <Card>
              <CardContent className="pt-6">
                <div className="space-y-2">
                  <div className="flex items-center justify-between text-sm">
                    <span>Uploading files...</span>
                    <span>{uploadProgress}%</span>
                  </div>
                  <Progress value={uploadProgress} />
                </div>
              </CardContent>
            </Card>
          )}

          {/* Actions */}
          <div className="flex items-center justify-between">
            <Button variant="outline" asChild>
              <Link href="/">Back to Overview</Link>
            </Button>
            <div className="flex gap-2">
              <Button
                onClick={handleUpload}
                disabled={!opticalFile || !thermalFile || uploading}
                className="bg-[#0E88D3] hover:bg-[#0E88D3]/90"
              >
                {uploading ? 'Uploading...' : 'Upload & Continue'}
                {!uploading && <ArrowRight className="ml-2 h-4 w-4" />}
              </Button>
            </div>
          </div>
        </div>
      </div>
    </DashboardLayout>
  );
}