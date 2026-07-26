if __name__ == '__main__':
    # usage: <script> total_tokens_in_B k_tok_per_s
    import sys
    total_tokens = int(sys.argv[1]) * 1e9
    k_tok_per_s = float(sys.argv[2]) * 1e3
    total_seconds = total_tokens / k_tok_per_s
    total_hours = total_seconds / 3600
    print(f"Total training time: {total_hours:.2f} hours")