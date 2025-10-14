'use client';

import { useState, useEffect } from 'react';
import { DashboardLayout } from '@/components/dashboard-layout';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { PlayCircle, Download, Trash2, RefreshCw, CheckCircle, XCircle, Clock } from 'lucide-react';
import { toast } from 'sonner';

export default function JobsPage() {
  const [jobs, setJobs] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Mock jobs data
    setTimeout(() => {
      setJobs([
        {
          id: '1',
          name: 'Urban Thermal SR - Downtown',
          status: 'completed',
          stage: 'evaluate',
          progress: 100,
          createdAt: '2025-06-15T10:30:00Z',
          updatedAt: '2025-06-15T10:35:00Z',
        },
        {
          id: '2',
          name: 'Forest Fire Detection',
          status: 'running',
          stage: 'fuse-sr',
          progress: 65,
          createdAt: '2025-06-15T11:00:00Z',
          updatedAt: '2025-06-15T11:10:00Z',
        },
        {
          id: '3',
          name: 'Agricultural Monitoring',
          status: 'queued',
          stage: 'align',
          progress: 0,
          createdAt: '2025-06-15T11:15:00Z',
          updatedAt: '2025-06-15T11:15:00Z',
        },
        {
          id: '4',
          name: 'Coastal Temperature Map',
          status: 'failed',
          stage: 'align',
          progress: 25,
          createdAt: '2025-06-15T09:45:00Z',
          updatedAt: '2025-06-15T09:50:00Z',
          errorMessage: 'CRS mismatch: unable to reproject thermal data',
        },
      ]);
      setLoading(false);
    }, 500);
  }, []);

  const getStatusIcon = (status) => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="h-4 w-4 text-green-500" />;
      case 'running':
        return <RefreshCw className="h-4 w-4 text-blue-500 animate-spin" />;
      case 'queued':
        return <Clock className="h-4 w-4 text-yellow-500" />;
      case 'failed':
        return <XCircle className="h-4 w-4 text-red-500" />;
      default:
        return null;
    }
  };

  const getStatusBadge = (status) => {
    const variants = {
      completed: 'default',
      running: 'secondary',
      queued: 'outline',
      failed: 'destructive',
    };
    return (
      <Badge variant={variants[status]} className="capitalize">
        {status}
      </Badge>
    );
  };

  const handleRerun = (jobId) => {
    toast.info(`Rerunning job ${jobId}...`);
  };

  const handleDownload = (jobId) => {
    toast.success(`Downloading artifacts for job ${jobId}...`);
  };

  const handleDelete = (jobId) => {
    toast.success(`Job ${jobId} deleted`);
    setJobs(jobs.filter(j => j.id !== jobId));
  };

  return (
    <DashboardLayout>
      <div className="container mx-auto px-4 py-8">
        <div className="max-w-7xl mx-auto space-y-8">
          {/* Header */}
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold mb-2">Job Queue</h1>
              <p className="text-muted-foreground">
                Monitor and manage your thermal SR processing jobs
              </p>
            </div>
            <Button className="bg-[#0E88D3] hover:bg-[#0E88D3]/90">
              <PlayCircle className="mr-2 h-4 w-4" />
              New Job
            </Button>
          </div>

          {/* Jobs Table */}
          <Card>
            <CardHeader>
              <CardTitle>All Jobs</CardTitle>
              <CardDescription>Recent processing jobs and their status</CardDescription>
            </CardHeader>
            <CardContent>
              {loading ? (
                <div className="flex items-center justify-center py-8">
                  <RefreshCw className="h-8 w-8 animate-spin text-muted-foreground" />
                </div>
              ) : (
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Job Name</TableHead>
                      <TableHead>Status</TableHead>
                      <TableHead>Stage</TableHead>
                      <TableHead>Progress</TableHead>
                      <TableHead>Created</TableHead>
                      <TableHead className="text-right">Actions</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {jobs.map((job) => (
                      <TableRow key={job.id}>
                        <TableCell className="font-medium">{job.name}</TableCell>
                        <TableCell>
                          <div className="flex items-center gap-2">
                            {getStatusIcon(job.status)}
                            {getStatusBadge(job.status)}
                          </div>
                        </TableCell>
                        <TableCell className="capitalize">{job.stage}</TableCell>
                        <TableCell>
                          <div className="flex items-center gap-2">
                            <Progress value={job.progress} className="w-24" />
                            <span className="text-sm text-muted-foreground">{job.progress}%</span>
                          </div>
                        </TableCell>
                        <TableCell className="text-sm text-muted-foreground">
                          {new Date(job.createdAt).toLocaleDateString()}
                        </TableCell>
                        <TableCell className="text-right">
                          <div className="flex items-center justify-end gap-2">
                            {job.status === 'completed' && (
                              <Button
                                size="sm"
                                variant="ghost"
                                onClick={() => handleDownload(job.id)}
                              >
                                <Download className="h-4 w-4" />
                              </Button>
                            )}
                            {(job.status === 'failed' || job.status === 'completed') && (
                              <Button
                                size="sm"
                                variant="ghost"
                                onClick={() => handleRerun(job.id)}
                              >
                                <RefreshCw className="h-4 w-4" />
                              </Button>
                            )}
                            <Button
                              size="sm"
                              variant="ghost"
                              onClick={() => handleDelete(job.id)}
                            >
                              <Trash2 className="h-4 w-4 text-red-500" />
                            </Button>
                          </div>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              )}
            </CardContent>
          </Card>

          {/* Summary Stats */}
          <div className="grid md:grid-cols-4 gap-4">
            <Card>
              <CardHeader className="pb-3">
                <CardDescription>Total Jobs</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="text-3xl font-bold">{jobs.length}</div>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="pb-3">
                <CardDescription>Completed</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="text-3xl font-bold text-green-500">
                  {jobs.filter(j => j.status === 'completed').length}
                </div>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="pb-3">
                <CardDescription>Running</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="text-3xl font-bold text-blue-500">
                  {jobs.filter(j => j.status === 'running').length}
                </div>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="pb-3">
                <CardDescription>Failed</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="text-3xl font-bold text-red-500">
                  {jobs.filter(j => j.status === 'failed').length}
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </DashboardLayout>
  );
}