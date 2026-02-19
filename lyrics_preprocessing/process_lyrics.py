import re
import json
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer

# ============================================
# PREPROCESSING FUNCTIONS
# ============================================

def preprocess_text(text):
    """Preprocess lyrics while preserving emotion signals"""
    original = text
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'\*{2,}', '[EXPLICIT]', text)
    text = text.lower()
    text = re.sub(r'\(\s*\)', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text, original

def extract_emotion_features(original_text):
    """Extract emotion-relevant features"""
    return {
        'has_exclamation': '!' in original_text,
        'has_question': '?' in original_text,
        'has_adlibs': bool(re.search(r'\([^)]*\)', original_text)),
        'has_caps': original_text != original_text.lower(),
        'word_count': len(original_text.split()),
        'char_count': len(original_text)
    }

# ============================================
# BERT TOKENIZATION
# ============================================

def initialize_tokenizer(model_name='bert-base-uncased'):
    """Initialize BERT tokenizer"""
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer

def tokenize_with_bert(text, tokenizer):
    """Apply BERT tokenization"""
    tokens = tokenizer.tokenize(text)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    attention_mask = [1] * len(token_ids)
    
    return {
        'tokens': tokens,
        'token_ids': token_ids,
        'attention_mask': attention_mask,
        'num_tokens': len(tokens)
    }

# ============================================
# LRC FILE PROCESSING
# ============================================

def process_single_lrc(filepath, tokenizer):
    """Process one LRC file"""
    metadata = {}
    lyrics_data = []
    
    metadata_pattern = r'\[([^:]+):([^\]]+)\]'
    timestamp_pattern = r'\[(\d{2}):(\d{2}\.\d{2})\](.*)'
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                meta_match = re.match(metadata_pattern, line)
                if meta_match and ':' in line:
                    key, value = meta_match.groups()
                    if not key.replace(':', '').replace('.', '').isdigit():
                        metadata[key] = value
                        continue
                
                ts_match = re.match(timestamp_pattern, line)
                if ts_match:
                    minutes, seconds, text = ts_match.groups()
                    timestamp_seconds = int(minutes) * 60 + float(seconds)
                    
                    if text.strip():
                        original_text = text.strip()
                        emotion_features = extract_emotion_features(original_text)
                        processed_text, _ = preprocess_text(original_text)
                        bert_tokens = tokenize_with_bert(processed_text, tokenizer)
                        
                        lyrics_data.append({
                            'timestamp': timestamp_seconds,
                            'original': original_text,
                            'processed': processed_text,
                            'bert_tokens': bert_tokens['tokens'],
                            'token_ids': bert_tokens['token_ids'],
                            'attention_mask': bert_tokens['attention_mask'],
                            'emotion_features': emotion_features
                        })
        
        return {
            'filename': filepath.stem,
            'metadata': metadata,
            'lyrics': lyrics_data,
            'summary': {
                'total_lines': len(lyrics_data),
                'total_bert_tokens': sum(len(l['bert_tokens']) for l in lyrics_data),
                'avg_tokens_per_line': sum(len(l['bert_tokens']) for l in lyrics_data) / len(lyrics_data) if lyrics_data else 0
            },
            'status': 'success'
        }
    
    except Exception as e:
        return {
            'filename': filepath.stem,
            'error': str(e),
            'status': 'failed'
        }

# ============================================
# BATCH PROCESSING
# ============================================

def batch_process_all_lrc_files(
    lrc_directory, 
    output_file='processed_lyrics_bert.json',
    checkpoint_interval=100,
    model_name='bert-base-uncased'
):
    """Process all LRC files"""
    
    tokenizer = initialize_tokenizer(model_name)
    lrc_files = sorted(list(Path(lrc_directory).glob('*.lrc')))
    total_files = len(lrc_files)
    
    print(f"\n{'='*50}")
    print(f"Found {total_files} LRC files")
    print(f"Output: {output_file}")
    print(f"Checkpoint interval: every {checkpoint_interval} files")
    print(f"{'='*50}\n")
    
    results = []
    errors = []
    
    for idx, lrc_file in enumerate(tqdm(lrc_files, desc="Processing")):
        result = process_single_lrc(lrc_file, tokenizer)
        
        if result['status'] == 'success':
            results.append(result)
        else:
            errors.append(result)
        
        if (idx + 1) % checkpoint_interval == 0:
            checkpoint_file = f"checkpoint_{idx+1}.json"
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\n→ Checkpoint saved: {checkpoint_file}")
    
    output_data = {
        'metadata': {
            'total_files': total_files,
            'successful': len(results),
            'failed': len(errors),
            'tokenizer_model': model_name
        },
        'data': results
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    if errors:
        with open('processing_errors.json', 'w', encoding='utf-8') as f:
            json.dump(errors, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*50}")
    print(f"PROCESSING COMPLETE")
    print(f"{'='*50}")
    print(f"✓ Successfully processed: {len(results)}/{total_files} files")
    print(f"✗ Failed: {len(errors)} files")
    print(f"→ Output saved to: {output_file}")
    if errors:
        print(f"→ Errors saved to: processing_errors.json")
    print(f"{'='*50}\n")
    
    return output_data

# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    
    # CHANGE THIS PATH TO YOUR LRC FILES FOLDER
    LRC_DIRECTORY = "/Users/devendram/Documents/CODING/PMEmo2019/lyrics"  # Put your .lrc files here
    OUTPUT_FILE = "processed_lyrics_bert.json"
    
    results = batch_process_all_lrc_files(
        lrc_directory=LRC_DIRECTORY,
        output_file=OUTPUT_FILE,
        checkpoint_interval=100,
        model_name='bert-base-uncased'
    )
    
    # Print sample
    print("\nSample output (first lyric line):")
    if results['data']:
        sample = results['data'][0]['lyrics'][0]
        print(f"Original: {sample['original']}")
        print(f"Processed: {sample['processed']}")
        print(f"BERT tokens: {sample['bert_tokens'][:10]}...")  # First 10 tokens
