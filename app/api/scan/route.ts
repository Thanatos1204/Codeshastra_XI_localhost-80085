import { NextRequest, NextResponse } from 'next/server';

// Assuming your Flask API is running at this URL
const API_URL = process.env.FLASK_API_URL || 'http://localhost:5000';

export async function POST(request: NextRequest) {
  try {
    // Forward the request to the Flask API
    const formData = await request.formData();
    
    const response = await fetch(`${API_URL}/api/scan`, {
      method: 'POST',
      body: formData,
    });
    
    if (!response.ok) {
      const errorText = await response.text();
      console.error('Error from Flask API:', errorText);
      return NextResponse.json({ error: 'Failed to process scan' }, { status: response.status });
    }
    
    const data = await response.json();
    
    // Transform the URLs to be relative to our Next.js app
    if (data.scans) {
      Object.keys(data.scans).forEach(key => {
        // Convert Flask API URL to our Next.js API URL
        // This assumes the Flask API returns URLs like /api/rooms/{room_id}/original
        data.scans[key] = `/api/proxy${data.scans[key]}`;
      });
    }
    
    return NextResponse.json(data);
  } catch (error) {
    console.error('Error processing scan request:', error);
    return NextResponse.json({ error: 'Internal server error' }, { status: 500 });
  }
}