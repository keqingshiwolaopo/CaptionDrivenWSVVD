## Project Structure

```
├── README.md               # Project documentation             
├── data/                   # Data storage directory
│   └── datasets/           
│       ├── xd/             
│       └── ucf/           
├── list/                   # Feature list directory
│   └── xd/                 
│       ├── rgb.list                           
│       ├── rgb_test.list           
│       ...      
│       └── gt.npy                  # Ground truth annotation
├── src/                    # Core code for model and training
├── ckpt/                   # Model checkpoint directory 
└── CE.py        
```

## 

## Requirements

```
python==3.7.13
torch==1.11.0  
cuda==11.3
numpy==1.21.5
```

## Training

```
python main.py 
```

## Testing

```
python main.py --eval --model_path ckpt/8352.pkl 
```

## Violent Video Caption

CE1 and CE2 captions for XD-Violence and UCF-Crime

https://data.mendeley.com/preview/kk3dyhr8jn?a=cc6e004d-9c55-43bd-b9b6-3c9f47a7b7a6

