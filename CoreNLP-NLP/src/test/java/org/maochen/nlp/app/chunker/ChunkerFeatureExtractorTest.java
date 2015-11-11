package org.maochen.nlp.app.chunker;

import org.junit.Test;

import static org.junit.Assert.assertNull;

/**
 * Created by Maochen on 11/10/15.
 */
public class ChunkerFeatureExtractorTest {

    @Test
    public void testNull() {
        assertNull(ChunkerFeatureExtractor.extract(null));
    }
}
