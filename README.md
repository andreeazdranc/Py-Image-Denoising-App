# Py-Image-Denoising-App

A Python-based application for removing noise from images using various denoising algorithms and techniques.

## Description

This application provides tools and algorithms for image denoising, helping to remove unwanted noise from digital images while preserving important details and features. It's designed to be easy to use for both beginners and advanced users working with noisy images.

## Features

- Multiple denoising algorithms implementation
- Support for various image formats
- Easy-to-use interface
- Batch processing capabilities
- Configurable parameters for fine-tuning results
- Visual comparison of original and denoised images

## Installation

### Prerequisites

- Python 3.7 or higher
- pip (Python package manager)

### Steps

1. Clone the repository:
```bash
git clone https://github.com/andreeazdranc/Py-Image-Denoising-App.git
cd Py-Image-Denoising-App
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
# Import the denoising module
from denoise import ImageDenoiser

# Create a denoiser instance
denoiser = ImageDenoiser()

# Load and denoise an image
denoised_image = denoiser.denoise('path/to/noisy/image.jpg')

# Save the result
denoised_image.save('path/to/output/image.jpg')
```

### Command Line Interface

```bash
python denoise.py --input input_image.jpg --output output_image.jpg --method gaussian
```

## Technologies Used

- **Python** - Core programming language
- **NumPy** - Numerical computing
- **OpenCV** - Image processing
- **Pillow** - Image manipulation
- **Matplotlib** - Visualization

## Project Structure

```
Py-Image-Denoising-App/
├── README.md
├── requirements.txt
├── denoise.py
├── utils/
│   └── image_utils.py
└── examples/
    └── sample_images/
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is open source and available under the [MIT License](LICENSE).

## Author

**Andreea Zdranc**
- GitHub: [@andreeazdranc](https://github.com/andreeazdranc)

## Acknowledgments

- Thanks to the open-source community for the various libraries used
- Inspired by classical and modern image denoising techniques