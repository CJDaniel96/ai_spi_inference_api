# Setting Up Actual Defect Images

## Overview
This guide explains how to replace the demo's placeholder images with your actual defect images for a more realistic demonstration.

## Directory Structure
Create the following folder structure in your `html` directory:

```
html/
├── ai_defect_detection_demo.html
├── AI_Defect_Detection_Demo_Guide.md
├── setup_defect_images.md (this file)
└── images/
    ├── defect1_surface_scratch.jpg
    ├── defect2_color_inconsistency.jpg
    ├── defect3_edge_damage.jpg
    ├── defect4_missing_component.jpg
    └── defect5_contamination_spot.jpg
```

## Image Requirements

### Format and Size
- **Format**: JPG, PNG, or WebP
- **Dimensions**: Recommended 400x300 pixels (or maintain 4:3 aspect ratio)
- **File Size**: Keep under 500KB each for fast loading
- **Quality**: High enough to show defect details clearly

### Defect Types to Prepare

1. **Surface Scratch** (`defect1_surface_scratch.jpg`)
   - Image showing linear scratches or abrasions
   - Clearly visible surface damage

2. **Color Inconsistency** (`defect2_color_inconsistency.jpg`)
   - Product with color variations or discoloration
   - Contrast should make the defect obvious

3. **Edge Damage** (`defect3_edge_damage.jpg`)
   - Chipped, cracked, or damaged edges
   - Structural integrity issues visible

4. **Missing Component** (`defect4_missing_component.jpg`)
   - Product with obviously missing parts
   - Empty spaces where components should be

5. **Contamination Spot** (`defect5_contamination_spot.jpg`)
   - Foreign material or contamination
   - Dirt, debris, or unwanted substances

## Updating the Demo Code

### Step 1: Create Images Folder
```bash
mkdir html/images
```

### Step 2: Add Your Images
Copy your 5 defect images to the `html/images/` folder with the exact names specified above.

### Step 3: Modify the HTML File
Replace the SVG placeholder images in `ai_defect_detection_demo.html`:

#### Find This Section (around line 200):
```javascript
const defectImages = [
    {
        id: 1,
        name: "Surface Scratch",
        image: "data:image/svg+xml;base64,PHN2ZyB3aWR0aD0i...",  // Long SVG data
        confidence: 94.2,
        description: "Linear scratch detected on product surface",
        defectArea: { x: 50, y: 100, width: 300, height: 100 }
    },
    // ... more entries
];
```

#### Replace With:
```javascript
const defectImages = [
    {
        id: 1,
        name: "Surface Scratch",
        image: "images/defect1_surface_scratch.jpg",
        confidence: 94.2,
        description: "Linear scratch detected on product surface",
        defectArea: { x: 50, y: 100, width: 300, height: 100 }
    },
    {
        id: 2,
        name: "Color Inconsistency",
        image: "images/defect2_color_inconsistency.jpg",
        confidence: 87.6,
        description: "Color variation outside acceptable tolerance",
        defectArea: { x: 120, y: 70, width: 160, height: 160 }
    },
    {
        id: 3,
        name: "Edge Damage",
        image: "images/defect3_edge_damage.jpg",
        confidence: 91.3,
        description: "Structural damage detected at product edge",
        defectArea: { x: 340, y: 80, width: 50, height: 40 }
    },
    {
        id: 4,
        name: "Missing Component",
        image: "images/defect4_missing_component.jpg",
        confidence: 96.8,
        description: "Expected component not found in designated area",
        defectArea: { x: 80, y: 80, width: 40, height: 40 }
    },
    {
        id: 5,
        name: "Contamination Spot",
        image: "images/defect5_contamination_spot.jpg",
        confidence: 89.4,
        description: "Foreign material contamination detected",
        defectArea: { x: 220, y: 100, width: 60, height: 40 }
    }
];
```

## Customizing Defect Information

### Adjusting Confidence Scores
Modify the `confidence` values (0-100) to reflect realistic AI detection accuracy:
```javascript
confidence: 94.2,  // Change to your desired percentage
```

### Updating Descriptions
Customize the `description` text to match your specific defect types:
```javascript
description: "Your custom defect description here",
```

### Adjusting Defect Areas
Modify `defectArea` coordinates to highlight the actual defect location in your images:
```javascript
defectArea: { x: 100, y: 150, width: 200, height: 100 }
// x, y: top-left corner of defect area
// width, height: size of defect area
```

## Advanced Customization

### Adding More Defect Types
```javascript
// Add new entries to the defectImages array
{
    id: 6,
    name: "Bubble Formation",
    image: "images/defect6_bubbles.jpg",
    confidence: 88.7,
    description: "Air bubbles detected in material surface",
    defectArea: { x: 150, y: 120, width: 80, height: 60 }
}
```

### Customizing Processing Times
```javascript
// Modify delay in processNextImage() function
setTimeout(() => {
    // Processing logic
}, Math.random() * 3000 + 2000); // 2-5 seconds processing time
```

### Adding Defect Severity Levels
```javascript
{
    id: 1,
    name: "Surface Scratch",
    image: "images/defect1_surface_scratch.jpg",
    confidence: 94.2,
    severity: "High", // Add severity level
    description: "Critical linear scratch detected on product surface",
    defectArea: { x: 50, y: 100, width: 300, height: 100 }
}
```

## Testing Your Setup

### Verification Checklist
- [ ] Images folder created in html directory
- [ ] 5 defect images copied with correct names
- [ ] HTML file updated with image paths
- [ ] Demo opens in browser without errors
- [ ] All images load correctly during demo
- [ ] Defect overlays align with actual defect locations

### Troubleshooting

#### Images Not Loading
- Check file paths are correct
- Ensure image files exist in the images folder
- Verify file names match exactly (case-sensitive)

#### Defect Overlays Misaligned
- Adjust `defectArea` coordinates in the JavaScript
- Use browser developer tools to find correct positions

#### Performance Issues
- Reduce image file sizes
- Optimize images for web (compress without losing quality)
- Consider using WebP format for better compression

## Example Workflow

1. **Collect Images**: Gather 5 high-quality defect images from your manufacturing process
2. **Prepare Images**: Resize to 400x300, optimize file sizes
3. **Name Files**: Use the exact naming convention specified
4. **Update Code**: Modify the HTML file to reference your images
5. **Test Demo**: Open in browser and verify everything works
6. **Fine-tune**: Adjust confidence scores, descriptions, and defect areas as needed

## Benefits of Using Real Images

- **Authenticity**: Demonstrates actual defect types from your process
- **Relevance**: Shows stakeholders real-world applications
- **Credibility**: Builds trust in the AI detection capabilities
- **Specificity**: Allows customization for your industry/products

Your AI defect detection demo is now ready with real manufacturing defect images!