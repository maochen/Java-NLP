/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 * 
 *   http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package classifier.backcup;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;


import opennlp.model.ComparableEvent;
import opennlp.model.Event;
import opennlp.model.EventStream;

/**
 * An indexer for maxent model data which handles cutoffs for uncommon contextual predicates and
 * provides a unique integer index for each of the predicates and maintains event values.
 * 
 * @author Tom Morton
 */
public class DataIndexer2 {
    private int numEvents;

    /** The predicate/context names. */
    protected String[] predLabels;

    /** The integer outcome associated with each unique event. */
    protected int[] outcomeList;

    /** The names of the outcomes. */
    protected String[] outcomeLabels;

    protected int[] numTimesEventsSeen;

    /** The integer contexts associated with each unique event. */
    int[][] contexts;

    /** The number of times each predicate occured. */
    protected int[] predCounts;

    float[][] values;

    /**
     * Updates the set of predicated and counter with the specified event contexts and cutoff. 
     * @param ec The contexts/features which occur in a event.
     * @param predicateSet The set of predicates which will be used for model building.
     * @param counter The predicate counters.
     * @param cutoff The cutoff which determines whether a predicate is included.
     */
     protected static void update(String[] ec, Set predicateSet, Map<String,Integer> counter, int cutoff) {
      for (int j=0; j<ec.length; j++) {
        Integer i = counter.get(ec[j]);
        if (i == null) {
          counter.put(ec[j], 1);
        }
        else {
          counter.put(ec[j], i+1);
        }
        if (!predicateSet.contains(ec[j]) && counter.get(ec[j]) >= cutoff) {
          predicateSet.add(ec[j]);
        }
      }
    }
    
    /**
     * Reads events from <tt>eventStream</tt> into a linked list.  The
     * predicates associated with each event are counted and any which
     * occur at least <tt>cutoff</tt> times are added to the
     * <tt>predicatesInOut</tt> map along with a unique integer index.
     *
     * @param eventStream an <code>EventStream</code> value
     * @param predicatesInOut a <code>TObjectIntHashMap</code> value
     * @param cutoff an <code>int</code> value
     * @return a <code>TLinkedList</code> value
     */
    private LinkedList<Event> computeEventCounts(RealBasicEventStream eventStream,Map<String,Integer> predicatesInOut,
        int cutoff) throws IOException {
      Set predicateSet = new HashSet();
      Map<String,Integer> counter = new HashMap<String,Integer>();
      LinkedList<Event> events = new LinkedList<Event>();
      while (eventStream.hasNext()) {
        Event ev = eventStream.next();
        events.addLast(ev);
        update(ev.getContext(),predicateSet,counter,cutoff);
      }
      predCounts = new int[predicateSet.size()];
      int index = 0;
      for (Iterator pi=predicateSet.iterator();pi.hasNext();index++) {
        String predicate = (String) pi.next();
        predCounts[index] = counter.get(predicate);
        predicatesInOut.put(predicate,index);
      }
      return events;
    }
    
    public DataIndexer2(RealBasicEventStream eventStream, int cutoff) throws IOException {
        Map<String, Integer> predicateIndex = new HashMap<String, Integer>();
        LinkedList<Event> events;
        List eventsToCompare;

        System.out.println("Indexing events using cutoff of " + cutoff + "\n");

        System.out.print("\tComputing event counts...  ");
        events = computeEventCounts(eventStream, predicateIndex, cutoff);
        System.out.println("done. " + events.size() + " events");

        System.out.print("\tIndexing...  ");
        eventsToCompare = index(events, predicateIndex);
        // done with event list
        events = null;
        // done with predicates
        predicateIndex = null;

        System.out.println("done.");

        System.out.print("Sorting and merging events... ");
        sortAndMerge(eventsToCompare, true);
        System.out.println("Done indexing.");    }


    public float[][] getValues() {
        return values;
    }

    protected int sortAndMerge(List eventsToCompare, boolean sort) {
        int numUniqueEvents = 1;
        numEvents = eventsToCompare.size();
        if (sort) {
            Collections.sort(eventsToCompare);
            if (numEvents <= 1) {
                return numUniqueEvents; // nothing to do; edge case (see assertion)
            }

            ComparableEvent ce = (ComparableEvent) eventsToCompare.get(0);
            for (int i = 1; i < numEvents; i++) {
                ComparableEvent ce2 = (ComparableEvent) eventsToCompare.get(i);

                if (ce.compareTo(ce2) == 0) {
                    ce.seen++; // increment the seen count
                    eventsToCompare.set(i, null); // kill the duplicate
                }
                else {
                    ce = ce2; // a new champion emerges...
                    numUniqueEvents++; // increment the # of unique events
                }
            }
        }
        else {
            numUniqueEvents = eventsToCompare.size();
        }
        if (sort) System.out.println("done. Reduced " + numEvents + " events to " + numUniqueEvents + ".");

        contexts = new int[numUniqueEvents][];
        outcomeList = new int[numUniqueEvents];
        numTimesEventsSeen = new int[numUniqueEvents];

        for (int i = 0, j = 0; i < numEvents; i++) {
            ComparableEvent evt = (ComparableEvent) eventsToCompare.get(i);
            if (null == evt) {
                continue; // this was a dupe, skip over it.
            }
            numTimesEventsSeen[j] = evt.seen;
            outcomeList[j] = evt.outcome;
            contexts[j] = evt.predIndexes;
            ++j;
        }

        values = new float[numUniqueEvents][];
        int numEvents = eventsToCompare.size();
        for (int i = 0, j = 0; i < numEvents; i++) {
            ComparableEvent evt = (ComparableEvent) eventsToCompare.get(i);
            if (null == evt) {
                continue; // this was a dupe, skip over it.
            }
            values[j++] = evt.values;
        }
        return numUniqueEvents;
    }

    protected List index(LinkedList<Event> events, Map<String, Integer> predicateIndex) {
        Map<String, Integer> omap = new HashMap<String, Integer>();

        int numEvents = events.size();
        int outcomeCount = 0;
        List eventsToCompare = new ArrayList(numEvents);
        List<Integer> indexedContext = new ArrayList<Integer>();

        for (int eventIndex = 0; eventIndex < numEvents; eventIndex++) {
            Event ev = (Event) events.removeFirst();
            String[] econtext = ev.getContext();
            ComparableEvent ce;

            int ocID;
            String oc = ev.getOutcome();

            if (omap.containsKey(oc)) {
                ocID = omap.get(oc);
            }
            else {
                ocID = outcomeCount++;
                omap.put(oc, ocID);
            }

            for (int i = 0; i < econtext.length; i++) {
                String pred = econtext[i];
                if (predicateIndex.containsKey(pred)) {
                    indexedContext.add(predicateIndex.get(pred));
                }
            }

            // drop events with no active features
            if (indexedContext.size() > 0) {
                int[] cons = new int[indexedContext.size()];
                for (int ci = 0; ci < cons.length; ci++) {
                    cons[ci] = indexedContext.get(ci);
                }
                ce = new ComparableEvent(ocID, cons, ev.getValues());
                eventsToCompare.add(ce);
            }
            else {
                System.err.println("Dropped event " + ev.getOutcome() + ":" + Arrays.asList(ev.getContext()));
            }
            // recycle the TIntArrayList
            indexedContext.clear();
        }
        outcomeLabels = toIndexedStringArray(omap);
        predLabels = toIndexedStringArray(predicateIndex);
        return eventsToCompare;
    }
    
    /**
    * Utility method for creating a String[] array from a map whose
    * keys are labels (Strings) to be stored in the array and whose
    * values are the indices (Integers) at which the corresponding
    * labels should be inserted.
    *
    * @param labelToIndexMap a <code>TObjectIntHashMap</code> value
    * @return a <code>String[]</code> value
    * @since maxent 1.2.6
    */
   protected static String[] toIndexedStringArray(Map<String,Integer> labelToIndexMap) {
     final String[] array = new String[labelToIndexMap.size()];
     for (String label : labelToIndexMap.keySet()) {
       array[labelToIndexMap.get(label)] = label;
     }
     return array;
   }

}
