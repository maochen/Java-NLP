package org.maochen.nlp.classifier.maxent.eventstream;

import opennlp.model.Event;
import opennlp.model.RealValueFileEventStream;

import org.apache.commons.lang3.StringUtils;

import java.util.Iterator;
import java.util.List;

/**
 * Created by Maochen on 12/10/14.
 */
public class StringEventStream implements EventStream, opennlp.model.EventStream {

    private Iterator<String[]> dataIter;

    // away pdiff=9.6875 ptwins=0.5 lose
    private Event createEvent(String obs) {
        int lastSpace = obs.lastIndexOf(StringUtils.SPACE);
        Event event = null;

        if (lastSpace != -1) {
            String label = obs.substring(lastSpace + 1);
            String[] contexts = obs.substring(0, lastSpace).split("\\s+");
            // Split name and value
            float[] values = RealValueFileEventStream.parseContexts(contexts);
            // Label, feature name, feature value
            event = new Event(label, contexts, values);
        }

        return event;
    }

    @Override
    public Event next() {
        String token = StringUtils.join(dataIter.next(), StringUtils.SPACE);
        return createEvent(token);
    }

    @Override
    public boolean hasNext() {
        return dataIter.hasNext();
    }

    public StringEventStream(List<String[]> data) {
        this.dataIter = data.iterator();
    }

}
