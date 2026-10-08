import argparse

from src.pipeline import run_all, run_task3_only


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the customer-review NLP pipeline.')
    parser.add_argument(
        '--task3-only',
        action='store_true',
        help='Regenerate Task 3 articles using the existing Task 2 cluster files.',
    )
    args = parser.parse_args()

    if args.task3_only:
        article = run_task3_only()
        print('Task 3 category articles generated:', len(article))
    else:
        results = run_all()
        print('Sentiment benchmark rows:', len(results['sentiment_comparison']))
        print('Clusters:', len(results['clusters']))
        print('Article saved:', len(results['article']))
