import { NextRequest, NextResponse } from 'next/server';

// Assuming your Flask API is running at this URL
const API_URL = process.env.FLASK_API_URL || 'http://localhost:5000';

export async function GET(
  request: NextRequest,
  { params }: { params: { roomId: string } }
) {
  const roomId = params.roomId;
  
  try {
    const response = await fetch(`${API_URL}/api/rooms/${roomId}`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    });
    
    if (!response.ok) {
      const errorText = await response.text();
      console.error('Error from Flask API:', errorText);
      return NextResponse.json({ error: 'Failed to fetch room' }, { status: response.status });
    }
    
    const data = await response.json();
    
    // Transform the URLs to be relative to our Next.js app
    if (data.scans) {
      Object.keys(data.scans).forEach(key => {
        // Convert Flask API URL to our Next.js API URL
        data.scans[key] = `/api/proxy${data.scans[key]}`;
      });
    }
    
    return NextResponse.json(data);
  } catch (error) {
    console.error('Error fetching room details:', error);
    return NextResponse.json({ error: 'Internal server error' }, { status: 500 });
  }
}