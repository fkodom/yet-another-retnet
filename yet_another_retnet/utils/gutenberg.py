import random
from typing import Generator, List, Literal, Optional

import requests
from torchdata.datapipes.iter import IterableWrapper, IterDataPipe

GUTENBERG_TOP_100_URLS = [
    # Romeo and Juliet by William Shakespeare
    "https://www.gutenberg.org/ebooks/1513.txt.utf-8",
    # Moby Dick; Or The Whale by Herman Melville
    "https://www.gutenberg.org/ebooks/2701.txt.utf-8",
    # A Room with a View by E. M. Forster
    "https://www.gutenberg.org/ebooks/2641.txt.utf-8",
    # Middlemarch by George Eliot
    "https://www.gutenberg.org/ebooks/145.txt.utf-8",
    # Little Women; Or Meg Jo Beth and Amy by Louisa May Alcott
    "https://www.gutenberg.org/ebooks/37106.txt.utf-8",
    # The Complete Works of William Shakespeare by William Shakespeare
    "https://www.gutenberg.org/ebooks/100.txt.utf-8",
    # The Blue Castle: a novel by L. M. Montgomery
    "https://www.gutenberg.org/ebooks/67979.txt.utf-8",
    # The Enchanted April by Elizabeth Von Arnim
    "https://www.gutenberg.org/ebooks/16389.txt.utf-8",
    # The Adventures of Ferdinand Count Fathom — Complete by T. Smollett
    "https://www.gutenberg.org/ebooks/6761.txt.utf-8",
    # Cranford by Elizabeth Cleghorn Gaskell
    "https://www.gutenberg.org/ebooks/394.txt.utf-8",
    # The Expedition of Humphry Clinker by T. Smollett
    "https://www.gutenberg.org/ebooks/2160.txt.utf-8",
    # The Adventures of Roderick Random by T. Smollett
    "https://www.gutenberg.org/ebooks/4085.txt.utf-8",
    # History of Tom Jones a Foundling by Henry Fielding
    "https://www.gutenberg.org/ebooks/6593.txt.utf-8",
    # Twenty Years After by Alexandre Dumas
    "https://www.gutenberg.org/ebooks/1259.txt.utf-8",
    # My Life — Volume 1 by Richard Wagner
    "https://www.gutenberg.org/ebooks/5197.txt.utf-8",
    # Pride and Prejudice by Jane Austen
    "https://www.gutenberg.org/ebooks/1342.txt.utf-8",
    # Down and Out in the Magic Kingdom by Cory Doctorow
    "https://www.gutenberg.org/ebooks/8086.txt.utf-8",
    # The Odyssey by Homer
    "https://www.gutenberg.org/ebooks/1727.txt.utf-8",
    # Alice's Adventures in Wonderland by Lewis Carroll
    "https://www.gutenberg.org/ebooks/11.txt.utf-8",
    # Frankenstein; Or The Modern Prometheus by Mary Wollstonecraft Shelley
    "https://www.gutenberg.org/ebooks/84.txt.utf-8",
    # Anna Karenina by graf Leo Tolstoy
    "https://www.gutenberg.org/ebooks/1399.txt.utf-8",
    # A Tale of Two Cities by Charles Dickens
    "https://www.gutenberg.org/ebooks/98.txt.utf-8",
    # Dracula by Bram Stoker
    "https://www.gutenberg.org/ebooks/345.txt.utf-8",
    # The Adventures of Sherlock Holmes by Arthur Conan Doyle
    "https://www.gutenberg.org/ebooks/1661.txt.utf-8",
    # The Picture of Dorian Gray by Oscar Wilde
    "https://www.gutenberg.org/ebooks/174.txt.utf-8",
    # The Count of Monte Cristo Illustrated by Alexandre Dumas
    "https://www.gutenberg.org/ebooks/1184.txt.utf-8",
    # The Brothers Karamazov by Fyodor Dostoyevsky
    "https://www.gutenberg.org/ebooks/28054.txt.utf-8",
    # War and Peace by graf Leo Tolstoy
    "https://www.gutenberg.org/ebooks/2600.txt.utf-8",
    # How to Pick a Mate: The Guide to a Happy Marriage by Clifford R. Adams and Vance Packard
    "https://www.gutenberg.org/ebooks/67472.txt.utf-8",
    # The Great Gatsby by F. Scott Fitzgerald
    "https://www.gutenberg.org/ebooks/64317.txt.utf-8",
    # Metamorphosis by Franz Kafka
    "https://www.gutenberg.org/ebooks/5200.txt.utf-8",
    # Ulysses by James Joyce
    "https://www.gutenberg.org/ebooks/4300.txt.utf-8",
    # The Prince by Niccolò Machiavelli
    "https://www.gutenberg.org/ebooks/1232.txt.utf-8",
    # Crime and Punishment by Fyodor Dostoyevsky
    "https://www.gutenberg.org/ebooks/2554.txt.utf-8",
    # The Yellow Wallpaper by Charlotte Perkins Gilman
    "https://www.gutenberg.org/ebooks/1952.txt.utf-8",
    # The Romance of Lust: A classic Victorian erotic novel by Anonymous
    "https://www.gutenberg.org/ebooks/30254.txt.utf-8",
    # The Kama Sutra of Vatsyayana by Vatsyayana
    "https://www.gutenberg.org/ebooks/27827.txt.utf-8",
    # Grimms' Fairy Tales by Jacob Grimm and Wilhelm Grimm
    "https://www.gutenberg.org/ebooks/2591.txt.utf-8",
    # A Modest Proposal by Jonathan Swift
    "https://www.gutenberg.org/ebooks/1080.txt.utf-8",
    # The Iliad by Homer
    "https://www.gutenberg.org/ebooks/6130.txt.utf-8",
    # Great Expectations by Charles Dickens
    "https://www.gutenberg.org/ebooks/1400.txt.utf-8",
    # Thus Spake Zarathustra: A Book for All and None by Friedrich Wilhelm Nietzsche
    "https://www.gutenberg.org/ebooks/1998.txt.utf-8",
    # The Adventures of Tom Sawyer Complete by Mark Twain
    "https://www.gutenberg.org/ebooks/74.txt.utf-8",
    # Meditations by Emperor of Rome Marcus Aurelius
    "https://www.gutenberg.org/ebooks/2680.txt.utf-8",
    # The Wonderful Wizard of Oz by L. Frank Baum
    "https://www.gutenberg.org/ebooks/55.txt.utf-8",
    # A Doll's House : a play by Henrik Ibsen
    "https://www.gutenberg.org/ebooks/2542.txt.utf-8",
    # The Importance of Being Earnest: A Trivial Comedy for Serious People by Oscar Wilde
    "https://www.gutenberg.org/ebooks/844.txt.utf-8",
    # Moby Multiple Language Lists of Common Words by Grady Ward
    "https://www.gutenberg.org/ebooks/3206.txt.utf-8",
    # Adventures of Huckleberry Finn by Mark Twain
    "https://www.gutenberg.org/ebooks/76.txt.utf-8",
    # The slang dictionary : by John Camden Hotten
    "https://www.gutenberg.org/ebooks/42108.txt.utf-8",
    # Jane Eyre: An Autobiography by Charlotte Brontë
    "https://www.gutenberg.org/ebooks/1260.txt.utf-8",
    # The Prophet by Kahlil Gibran
    "https://www.gutenberg.org/ebooks/58585.txt.utf-8",
    # Beyond Good and Evil by Friedrich Wilhelm Nietzsche
    "https://www.gutenberg.org/ebooks/4363.txt.utf-8",
    # Anne of Green Gables by L. M. Montgomery
    "https://www.gutenberg.org/ebooks/45.txt.utf-8",
    # On the Duty of Civil Disobedience by Henry David Thoreau
    "https://www.gutenberg.org/ebooks/71.txt.utf-8",
    # Tractatus Logico-Philosophicus by Ludwig Wittgenstein
    "https://www.gutenberg.org/ebooks/5740.txt.utf-8",
    # The Strange Case of Dr. Jekyll and Mr. Hyde by Robert Louis Stevenson
    "https://www.gutenberg.org/ebooks/43.txt.utf-8",
    # Don Quixote by Miguel de Cervantes Saavedra
    "https://www.gutenberg.org/ebooks/996.txt.utf-8",
    # Calculus Made Easy by Silvanus P. Thompson
    "https://www.gutenberg.org/ebooks/33283.txt.utf-8",
    # Mark Twain's Speeches by Mark Twain
    "https://www.gutenberg.org/ebooks/3188.txt.utf-8",
    # Treasure Island by Robert Louis Stevenson
    "https://www.gutenberg.org/ebooks/120.txt.utf-8",
    # The Rámáyan of Válmíki translated into English verse by Valmiki
    "https://www.gutenberg.org/ebooks/24869.txt.utf-8",
    # A Child's History of the World by V. M. Hillyer
    "https://www.gutenberg.org/ebooks/67149.txt.utf-8",
    # A Study in Scarlet by Arthur Conan Doyle
    "https://www.gutenberg.org/ebooks/244.txt.utf-8",
    # Walden and On The Duty Of Civil Disobedience by Henry David Thoreau
    "https://www.gutenberg.org/ebooks/205.txt.utf-8",
    # Wuthering Heights by Emily Brontë
    "https://www.gutenberg.org/ebooks/768.txt.utf-8",
    # The Republic by Plato
    "https://www.gutenberg.org/ebooks/1497.txt.utf-8",
    # Essays of Michel de Montaigne — Complete by Michel de Montaigne
    "https://www.gutenberg.org/ebooks/3600.txt.utf-8",
    # Winnie-the-Pooh by A. A. Milne
    "https://www.gutenberg.org/ebooks/67098.txt.utf-8",
    # Emma by Jane Austen
    "https://www.gutenberg.org/ebooks/158.txt.utf-8",
    # The divine comedy by Dante Alighieri
    "https://www.gutenberg.org/ebooks/8800.txt.utf-8",
    # The Scarlet Letter by Nathaniel Hawthorne
    "https://www.gutenberg.org/ebooks/25344.txt.utf-8",
    # Little Women by Louisa May Alcott
    "https://www.gutenberg.org/ebooks/514.txt.utf-8",
    # Les Misérables by Victor Hugo
    "https://www.gutenberg.org/ebooks/135.txt.utf-8",
    # Demonology and Devil-lore by Moncure Daniel Conway
    "https://www.gutenberg.org/ebooks/40686.txt.utf-8",
    # Spoon River Anthology by Edgar Lee Masters
    "https://www.gutenberg.org/ebooks/1280.txt.utf-8",
    # The King James Version of the Bible
    "https://www.gutenberg.org/ebooks/10.txt.utf-8",
    # Peter Pan by J. M. Barrie
    "https://www.gutenberg.org/ebooks/16.txt.utf-8",
    # The King in Yellow by Robert W. Chambers
    "https://www.gutenberg.org/ebooks/8492.txt.utf-8",
    # The War of the Worlds by H. G. Wells
    "https://www.gutenberg.org/ebooks/36.txt.utf-8",
    # Notes from the Underground by Fyodor Dostoyevsky
    "https://www.gutenberg.org/ebooks/600.txt.utf-8",
    # The Problems of Philosophy by Bertrand Russell
    "https://www.gutenberg.org/ebooks/5827.txt.utf-8",
    # Heart of Darkness by Joseph Conrad
    "https://www.gutenberg.org/ebooks/219.txt.utf-8",
    # The Time Machine by H. G. Wells
    "https://www.gutenberg.org/ebooks/35.txt.utf-8",
    # A Christmas Carol in Prose; Being a Ghost Story of Christmas by Charles Dickens
    "https://www.gutenberg.org/ebooks/46.txt.utf-8",
    # Carmilla by Joseph Sheridan Le Fanu
    "https://www.gutenberg.org/ebooks/10007.txt.utf-8",
    # Dubliners by James Joyce
    "https://www.gutenberg.org/ebooks/2814.txt.utf-8",
    # David Copperfield by Charles Dickens
    "https://www.gutenberg.org/ebooks/766.txt.utf-8",
    # Sense and Sensibility by Jane Austen
    "https://www.gutenberg.org/ebooks/161.txt.utf-8",
    # Dombey and Son by Charles Dickens
    "https://www.gutenberg.org/ebooks/821.txt.utf-8",
    # The Jungle Book by Rudyard Kipling
    "https://www.gutenberg.org/ebooks/236.txt.utf-8",
    # The Art of War by active 6th century B.C. Sunzi
    "https://www.gutenberg.org/ebooks/132.txt.utf-8",
    # The murder of Roger Ackroyd by Agatha Christie
    "https://www.gutenberg.org/ebooks/69087.txt.utf-8",
    # The Souls of Black Folk by W. E. B. Du Bois
    "https://www.gutenberg.org/ebooks/408.txt.utf-8",
    # The Marching Morons by C. M. Kornbluth
    "https://www.gutenberg.org/ebooks/51233.txt.utf-8",
    # Josefine Mutzenbacher by Felix Salten
    "https://www.gutenberg.org/ebooks/31284.txt.utf-8",
    # Oliver Twist by Charles Dickens
    "https://www.gutenberg.org/ebooks/730.txt.utf-8",
    # Through the Looking-Glass by Lewis Carroll
    "https://www.gutenberg.org/ebooks/12.txt.utf-8",
    # The Confessions of St. Augustine by Bishop of Hippo Saint Augustine
    "https://www.gutenberg.org/ebooks/3296.txt.utf-8",
    # Struwwelpeter: Merry Stories and Funny Pictures by Heinrich Hoffmann
    "https://www.gutenberg.org/ebooks/12116.txt.utf-8",
]


def get_split_indices(
    num_samples: int,
    split: Literal["train", "val", "test"],
    seed: int = 42,
    val_split: float = 0.1,
    test_split: float = 0.1,
) -> List[int]:
    indices = list(range(num_samples))
    random.seed(seed)
    random.shuffle(indices)

    num_val = int(num_samples * val_split)
    num_test = int(num_samples * test_split)

    if split == "train":
        out = indices[num_val + num_test :]
    elif split == "val":
        out = indices[:num_val]
    elif split == "test":
        out = indices[num_val : num_val + num_test]
    else:
        raise ValueError(f"Invalid split: {split}")

    return sorted(out)


class GutenbergEBookLoader(IterDataPipe[str]):
    def __init__(self, dp: IterDataPipe[str]):
        self.dp = dp

    def __iter__(self) -> Generator[str, None, None]:
        for url in self.dp:
            resp = requests.get(url)
            try:
                resp.raise_for_status()
            except requests.exceptions.HTTPError:
                continue

            text = resp.text
            text = text.split("***", maxsplit=2)[-1]  # remove header
            text = text.split("*** End", maxsplit=1)[0]  # remove footer
            text = text.strip()

            yield text


class TextChunker(IterDataPipe[str]):
    def __init__(
        self,
        dp: IterDataPipe[str],
        chunk_size: int = 4096,
        step_size: Optional[int] = None,
        drop_last: bool = False,
    ):
        self.dp = dp
        self.chunk_size = chunk_size
        self.step_size = step_size or chunk_size
        self.drop_last = drop_last

    def __iter__(self) -> Generator[str, None, None]:
        for text in self.dp:
            for i in range(0, len(text), self.step_size):
                chunk = text[i : i + self.chunk_size]
                if self.drop_last and len(chunk) < self.chunk_size:
                    continue

                chunk = chunk.split(" ", maxsplit=1)[-1]  # leading partial words
                chunk = chunk.rsplit(" ", maxsplit=1)[0]  # trailing partial words
                chunk = chunk.strip()  # leading/trailing whitespace
                yield chunk


def project_gutenberg_top_100_datapipe(
    split: Literal["train", "val", "test"],
    chunk_size: int = 4096,
    step_size: Optional[int] = None,
    shuffle: bool = False,
    shuffle_buffer_size: int = 8192,
    drop_last: bool = True,
) -> IterDataPipe[str]:
    indices = get_split_indices(len(GUTENBERG_TOP_100_URLS), split=split)
    if shuffle:
        random.shuffle(indices)

    # Iterable datapipe of ebook URLs (ebooks are UTF-8 text files)
    pipe: IterDataPipe = IterableWrapper([GUTENBERG_TOP_100_URLS[i] for i in indices])
    pipe = GutenbergEBookLoader(pipe)
    pipe = TextChunker(
        pipe, chunk_size=chunk_size, step_size=step_size, drop_last=drop_last
    )
    pipe = pipe.sharding_filter()
    if shuffle:
        pipe = pipe.shuffle(buffer_size=shuffle_buffer_size)

    return pipe
