import unittest

from music_domain import (
    MUSIC_CATALOG,
    encode_track,
    format_track_card,
    pick_next_track,
    recommendation_percentage,
)


class MusicDomainTests(unittest.TestCase):
    def test_catalog_titles_are_unique(self):
        titles = [track.title for track in MUSIC_CATALOG]
        self.assertEqual(len(titles), len(set(titles)))

    def test_encode_track_returns_eleven_binary_features(self):
        bits = encode_track(MUSIC_CATALOG[0], "Focus")
        self.assertEqual(len(bits), 11)
        self.assertTrue(all(bit in (0, 1) for bit in bits))

    def test_encode_track_is_deterministic(self):
        track = MUSIC_CATALOG[3]
        self.assertEqual(
            encode_track(track, "Workout"),
            encode_track(track, "Workout"),
        )

    def test_unknown_context_uses_default_bucket(self):
        track = MUSIC_CATALOG[1]
        self.assertEqual(encode_track(track, "not a context")[-2:], [0, 0])

    def test_recommendation_percentage(self):
        self.assertEqual(recommendation_percentage([1, 1, 0, 0]), 50)
        self.assertEqual(recommendation_percentage([]), 0)

    def test_pick_next_track_skips_seen_titles(self):
        first = MUSIC_CATALOG[0]
        picked = pick_next_track([first.title], 0)
        self.assertNotEqual(picked.title, first.title)

    def test_format_track_card_contains_key_fields(self):
        track = MUSIC_CATALOG[0]
        card = format_track_card(track)
        self.assertIn(track.title, card)
        self.assertIn(track.artist, card)
        self.assertIn(str(track.tempo_bpm), card)


if __name__ == "__main__":
    unittest.main()
