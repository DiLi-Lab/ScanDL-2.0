

from scandl2_pkg.scandl2 import ScanDL2


def main():
    
    sample_sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "She enjoys reading books in her free time.",
        "Artificial intelligence is transforming the world.",
        "The weather today is sunny with a light breeze.",
    ]
    sample_paragraphs = [
        "The sun was shining brightly in the clear blue sky. Birds chirped melodiously from the trees, filling the air with a cheerful tune. It was the perfect day for a walk in the park.",
        "She had always loved the scent of fresh coffee in the morning. As she sipped her espresso, she glanced at the newspaper headlines. Another day, another adventure awaited her.",
        "The conference room was filled with eager attendees. The keynote speaker took the stage, adjusting the microphone. With a confident smile, he began his presentation on the future of artificial intelligence.",
        "They packed their bags and set off on their long-awaited road trip. The open road stretched ahead, inviting them to explore new places. With music playing softly in the background, they felt a sense of freedom like never before.",
    ]

    scandl2_sent = ScanDL2(
        text_type='sentence',
        bsz=2,
        save='scandl2_pkg/test_sentences',
        filename='sample_output',
    )
    out_sent = scandl2_sent(sample_sentences)
    
    scandl2_par = ScanDL2(
        text_type='paragraph',
        bsz=2,
        save='scandl2_pkg/test_paragraphs',
        filename='sample_output',
    )

    out_par = scandl2_par(sample_paragraphs)


if __name__ == '__main__':
    main() 
