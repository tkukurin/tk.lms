from tk.utils.utils import fetch


def tinyshakespeare() -> str:
    url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    return fetch(url).decode('utf8')


def words(lang: str = 'en') -> set[str]:
    assert lang in ('en', )
    url = 'https://github.com/dwyl/english-words/raw/master/words.txt'
    text = fetch(url).decode('utf8')
    return {x for x in text.split('\n') if x.strip()}
