# Interrupt System for Hand Gesture Recognition

## Overview

The interrupt system allows you to immediately switch from normal gesture sequence processing to direct dog locomotion control when a closed fist is detected. This acts as a "panic button" that drops all queued actions and takes immediate control.

## How It Works

### 1. **Normal Mode** (Default)
- Right hand gestures are processed as sequences
- Gestures are queued and interpreted by the game
- Left hand can be used for other purposes

### 2. **Interrupt Mode** (Triggered by Closed Fist)
- Left hand closed fist triggers interrupt
- All queued gesture sequences are cleared
- Right hand gestures become direct dog control commands
- Immediate response, no queuing

### 3. **Mode Switching**
- **Enter Interrupt**: Left hand closes → Immediate switch to dog control
- **Exit Interrupt**: Left hand opens → Return to normal gesture processing

## Server Endpoints

### Interrupt Control
- `POST /trigger_interrupt` - Activate interrupt mode
- `POST /clear_interrupt` - Deactivate interrupt mode  
- `GET /get_interrupt_status` - Check current interrupt status

### Data Endpoints
- `POST /upload_Fingersequence` - Normal gesture sequences (ignored during interrupt)
- `POST /dog_control` - Direct dog control commands (only active during interrupt)
- `GET /get_Fingersequence` - Get gesture history

## Integration with Games

### Basic Integration Pattern

```python
import requests

class GameController:
    def __init__(self):
        self.server_url = "http://127.0.0.1:50007"
        self.interrupt_active = False
    
    def check_interrupt_status(self):
        """Monitor for interrupt state changes."""
        response = requests.get(f"{self.server_url}/get_interrupt_status")
        if response.status_code == 200:
            return response.json()["interrupt_active"]
        return False
    
    def handle_mode_switch(self, new_interrupt_state):
        """Handle switching between normal and interrupt modes."""
        if new_interrupt_state and not self.interrupt_active:
            # Entering interrupt mode
            self.interrupt_active = True
            self.clear_gesture_queue()
            self.switch_to_dog_control()
        elif not new_interrupt_state and self.interrupt_active:
            # Exiting interrupt mode
            self.interrupt_active = False
            self.resume_normal_processing()
```

### Game-Specific Implementation

```python
def switch_to_dog_control(self):
    """Switch game to dog control mode."""
    # Pause current game actions
    self.pause_current_actions()
    
    # Clear any queued commands
    self.clear_command_queue()
    
    # Switch input handling to direct dog control
    self.set_input_mode("dog_control")
    
    print("🚨 Switched to dog control mode!")

def resume_normal_processing(self):
    """Resume normal gesture processing."""
    # Resume normal gameplay
    self.resume_gameplay()
    
    # Restart gesture sequence processing
    self.restart_gesture_processing()
    
    print("✅ Resumed normal gesture processing!")
```

## Usage Examples

### 1. Test the System
```bash
# Test interrupt endpoints
python test_interrupt_system.py test

# Run game controller simulation
python test_interrupt_system.py
```

### 2. Start the System
```bash
# Terminal 1: Start the gesture server
python gestureServer.py

# Terminal 2: Start the hand gesture recognition
python app.py

# Terminal 3: Start your game (integrate with the server)
```

## Visual Feedback

The system provides visual feedback on the camera feed:
- **Green text**: "Normal Mode - Gesture Sequences"
- **Red text**: "🚨 INTERRUPT MODE - Dog Control Active"
- **Console messages**: Real-time status updates

## Data Flow

### Normal Mode
```
Right Hand Gesture → Point History → Classifier → Upload Sequence → Game Processes
```

### Interrupt Mode  
```
Right Hand Gesture → Point History → Classifier → Direct Dog Command → Immediate Execution
```

## Key Features

1. **Immediate Response**: No queuing during interrupt mode
2. **State Persistence**: Interrupt remains active until explicitly cleared
3. **Data Isolation**: Normal sequences are ignored during interrupt
4. **Visual Feedback**: Clear indication of current mode
5. **Thread Safety**: Proper locking for concurrent access

## Troubleshooting

### Common Issues

1. **Interrupt not triggering**
   - Ensure left hand is properly detected as "Close" gesture
   - Check server is running on correct port (50007)

2. **Mode not switching**
   - Verify interrupt status endpoint is responding
   - Check network connectivity between client and server

3. **Commands not executing**
   - Ensure you're using the correct endpoint for current mode
   - Check that interrupt is active for dog control commands

### Debug Commands

```python
# Check server status
curl http://127.0.0.1:50007/get_interrupt_status

# Manually trigger interrupt
curl -X POST http://127.0.0.1:50007/trigger_interrupt

# Clear interrupt
curl -X POST http://127.0.0.1:50007/clear_interrupt
```

## Integration Checklist

- [ ] Server running on port 50007
- [ ] Client sending interrupt signals on left hand close
- [ ] Game monitoring interrupt status
- [ ] Game handling mode switching
- [ ] Game processing appropriate commands for each mode
- [ ] Visual feedback implemented
- [ ] Error handling for network issues
- [ ] Testing with both hands in frame

This interrupt system provides a robust way to implement immediate control switching in gesture-based applications, perfect for games that need both sequence-based and direct control modes. 