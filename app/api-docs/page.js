'use client';

import { DashboardLayout } from '@/components/dashboard-layout';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Code } from 'lucide-react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';

export default function ApiDocsPage() {
  const endpoints = [
    {
      method: 'POST',
      path: '/api/upload',
      description: 'Upload optical and thermal files',
      request: `{
  "optical": File,
  "thermal": File
}`,
      response: `{
  "success": true,
  "jobId": "uuid",
  "message": "Files uploaded successfully"
}`,
    },
    {
      method: 'POST',
      path: '/api/jobs/{jobId}/align',
      description: 'Start alignment process for a job',
      request: `{
  "method": "orb",
  "matchThreshold": 0.75,
  "ransacThreshold": 3.0
}`,
      response: `{
  "success": true,
  "status": "running",
  "progress": 10
}`,
    },
    {
      method: 'POST',
      path: '/api/jobs/{jobId}/sr',
      description: 'Start super-resolution process',
      request: `{
  "model": "cnn",
  "scale": 2,
  "edgeWeight": 0.7,
  "textureWeight": 0.5
}`,
      response: `{
  "success": true,
  "status": "running",
  "progress": 15
}`,
    },
    {
      method: 'GET',
      path: '/api/jobs/{jobId}',
      description: 'Get job status and details',
      response: `{
  "id": "uuid",
  "name": "Job name",
  "status": "completed",
  "progress": 100,
  "stage": "evaluate",
  "config": {...},
  "artifacts": {
    "aligned": "/path/to/aligned.tif",
    "sr": "/path/to/sr.tif",
    "metrics": "/path/to/metrics.csv"
  }
}`,
    },
    {
      method: 'GET',
      path: '/api/jobs/{jobId}/metrics',
      description: 'Get evaluation metrics for a job',
      response: `{
  "psnr": 28.42,
  "ssim": 0.891,
  "rmse": 1.24,
  "perClass": [...]
}`,
    },
    {
      method: 'GET',
      path: '/api/jobs',
      description: 'List all jobs for current user',
      response: `{
  "jobs": [
    {
      "id": "uuid",
      "name": "Job name",
      "status": "completed",
      "progress": 100,
      "createdAt": "2025-06-15T10:30:00Z"
    }
  ],
  "total": 10
}`,
    },
  ];

  return (
    <DashboardLayout>
      <div className="container mx-auto px-4 py-8">
        <div className="max-w-6xl mx-auto space-y-8">
          {/* Header */}
          <div>
            <h1 className="text-3xl font-bold mb-2 flex items-center gap-2">
              <Code className="h-8 w-8 text-[#0E88D3]" />
              API Documentation
            </h1>
            <p className="text-muted-foreground">
              REST API endpoints for programmatic access to the thermal SR pipeline
            </p>
          </div>

          {/* Authentication */}
          <Card>
            <CardHeader>
              <CardTitle>Authentication</CardTitle>
              <CardDescription>All API requests require authentication</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <p className="text-sm text-muted-foreground">
                API requests are authenticated using session cookies. Login via <code className="bg-muted px-1 py-0.5 rounded">/api/auth/login</code> to obtain a session.
              </p>
              <div className="bg-muted p-4 rounded-lg">
                <code className="text-sm">
                  curl -X POST https://thermal-sr-optics.preview.emergentagent.com/api/auth/login \
                  <br />
                  {"  "}-H "Content-Type: application/json" \
                  <br />
                  {"  "}-d '{"email": "user@example.com", "password": "password"}' \
                  <br />
                  {"  "}-c cookies.txt
                </code>
              </div>
            </CardContent>
          </Card>

          {/* Endpoints */}
          <div className="space-y-4">
            <h2 className="text-2xl font-bold">Endpoints</h2>
            {endpoints.map((endpoint, index) => (
              <Card key={index}>
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <CardTitle className="text-lg flex items-center gap-2">
                      <Badge
                        variant={endpoint.method === 'GET' ? 'outline' : 'default'}
                        className={endpoint.method === 'POST' ? 'bg-[#0E88D3]' : ''}
                      >
                        {endpoint.method}
                      </Badge>
                      <code className="text-sm">{endpoint.path}</code>
                    </CardTitle>
                  </div>
                  <CardDescription>{endpoint.description}</CardDescription>
                </CardHeader>
                <CardContent>
                  <Tabs defaultValue="response" className="w-full">
                    <TabsList>
                      {endpoint.request && <TabsTrigger value="request">Request</TabsTrigger>}
                      <TabsTrigger value="response">Response</TabsTrigger>
                      <TabsTrigger value="curl">cURL</TabsTrigger>
                    </TabsList>
                    {endpoint.request && (
                      <TabsContent value="request">
                        <div className="bg-muted p-4 rounded-lg">
                          <pre className="text-sm overflow-x-auto">
                            <code>{endpoint.request}</code>
                          </pre>
                        </div>
                      </TabsContent>
                    )}
                    <TabsContent value="response">
                      <div className="bg-muted p-4 rounded-lg">
                        <pre className="text-sm overflow-x-auto">
                          <code>{endpoint.response}</code>
                        </pre>
                      </div>
                    </TabsContent>
                    <TabsContent value="curl">
                      <div className="bg-muted p-4 rounded-lg">
                        <code className="text-sm">
                          curl -X {endpoint.method}{' '}
                          https://thermal-sr-optics.preview.emergentagent.com{endpoint.path}{' '}\
                          <br />
                          {"  "}-H "Content-Type: application/json" \
                          <br />
                          {"  "}-b cookies.txt
                          {endpoint.request && (
                            <>
                              {' '}\
                              <br />
                              {"  "}-d '{JSON.stringify(JSON.parse(endpoint.request), null, 2).replace(/\n/g, '')}'
                            </>
                          )}
                        </code>
                      </div>
                    </TabsContent>
                  </Tabs>
                </CardContent>
              </Card>
            ))}
          </div>

          {/* Rate Limits */}
          <Card>
            <CardHeader>
              <CardTitle>Rate Limits</CardTitle>
              <CardDescription>API usage limits</CardDescription>
            </CardHeader>
            <CardContent>
              <ul className="space-y-2 text-sm text-muted-foreground">
                <li>• 100 requests per minute per user</li>
                <li>• 10 concurrent SR jobs per user</li>
                <li>• Maximum file size: 500 MB per file</li>
              </ul>
            </CardContent>
          </Card>
        </div>
      </div>
    </DashboardLayout>
  );
}