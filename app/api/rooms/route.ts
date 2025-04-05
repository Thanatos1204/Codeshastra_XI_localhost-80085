import { NextRequest, NextResponse } from 'next/server';

// Assuming your Flask API is running at this URL
const API_URL = process.env.FLASK_API_URL || 'http://localhost:5000';

export async function GET(request: NextRequest) {
  try {
    const response = await fetch(`${API_URL}/api/rooms`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    });
    
    if (!response.ok) {
      const errorText = await response.text();
      console.error('Error from Flask API:', errorText);
      return NextResponse.json({ error: 'Failed to fetch rooms' }, { status: response.status });
    }
    
    const data = await response.json();
    
    // Transform the URLs to be relative to our Next.js app
    if (data.rooms) {
      data.rooms.forEach((room: any) => {
        if (room.thumbnail) {
          // Convert Flask API URL to our Next.js API URL
          room.thumbnail = `/api/proxy${room.thumbnail}`;
        }
      });
    }
    
    return NextResponse.json(data);
  } catch (error) {
    console.error('Error fetching rooms:', error);
    return NextResponse.json({ error: 'Internal server error' }, { status: 500 });
  }
}