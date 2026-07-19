// ============================================================================
// Illuminations — scripture embedded through Vellum.
// Curated around creation, light, craft and making, so they read as part of
// the illuminated-manuscript concept rather than decoration bolted on.
// ============================================================================

export interface Verse {
  text: string;
  ref: string;
}

/** Shown while the model weights download / warm up — themes of beginning & light. */
export const LOADING_VERSES: Verse[] = [
  { text: "In the beginning God created the heaven and the earth.", ref: "Genesis 1:1" },
  { text: "And God said, Let there be light: and there was light.", ref: "Genesis 1:3" },
  { text: "And God saw the light, that it was good.", ref: "Genesis 1:4" },
  { text: "The earth was without form, and void; and darkness was upon the face of the deep.", ref: "Genesis 1:2" },
  { text: "Thy word is a lamp unto my feet, and a light unto my path.", ref: "Psalm 119:105" },
  { text: "For with thee is the fountain of life: in thy light shall we see light.", ref: "Psalm 36:9" },
];

/** Shown during generation — themes of forming, making, hands at work. */
export const GENERATING_VERSES: Verse[] = [
  { text: "But now, O LORD, thou art our father; we are the clay, and thou our potter.", ref: "Isaiah 64:8" },
  { text: "He hath made every thing beautiful in his time.", ref: "Ecclesiastes 3:11" },
  { text: "Behold, I make all things new.", ref: "Revelation 21:5" },
  { text: "And let the beauty of the LORD our God be upon us: and establish thou the work of our hands.", ref: "Psalm 90:17" },
  { text: "I will praise thee; for I am fearfully and wonderfully made.", ref: "Psalm 139:14" },
  { text: "Every good gift and every perfect gift is from above.", ref: "James 1:17" },
];

/** Quiet footer rotation — themes of vision, horizon, the unseen expanse. */
export const AMBIENT_VERSES: Verse[] = [
  { text: "Lift up thine eyes, and look from the place where thou art.", ref: "Genesis 13:14" },
  { text: "He stretcheth out the north over the empty place, and hangeth the earth upon nothing.", ref: "Job 26:7" },
  { text: "The heavens declare the glory of God; and the firmament sheweth his handywork.", ref: "Psalm 19:1" },
  { text: "Eye hath not seen, nor ear heard... the things which God hath prepared.", ref: "1 Corinthians 2:9" },
  { text: "For now we see through a glass, darkly; but then face to face.", ref: "1 Corinthians 13:12" },
  { text: "Enlarge the place of thy tent... lengthen thy cords, and strengthen thy stakes.", ref: "Isaiah 54:2" },
];

