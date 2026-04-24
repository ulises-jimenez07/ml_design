import argparse
import json
import logging
import os
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions, StandardOptions, SetupOptions
from apache_beam.transforms import window

# Assuming Vertex AI Feature Store (Next Gen) which is backed by BigQuery.
# For real-time updates, we stream data into the BigQuery table that backs the Feature View.

class ParseEvent(beam.DoFn):
    def process(self, element):
        try:
            record = json.loads(element.decode('utf-8'))
            # Expected schema: {"event_id": "str", "user_id": "str", "timestamp": "ISO8601"}
            if 'event_id' in record:
                yield (record['event_id'], 1)
        except Exception as e:
            logging.error(f"Error parsing event: {e}")

class FormatForBigQuery(beam.DoFn):
    def process(self, element, window=beam.DoFn.WindowParam):
        event_id, count = element
        # Feature Store requires an entity ID and feature values, plus a timestamp
        yield {
            'event_id': event_id,
            'click_count_5m': count,
            'feature_timestamp': window.end.to_rfc3339()
        }

def run(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_subscription', 
                        default=os.environ.get('PUBSUB_SUBSCRIPTION', 'projects/YOUR_PROJECT/subscriptions/event-sub'),
                        help='Input Pub/Sub subscription.')
    parser.add_argument('--output_table', 
                        default=os.environ.get('BQ_FEATURE_TABLE', 'YOUR_PROJECT:feature_store.event_features_rt'),
                        help='Output BigQuery table for Feature Store.')
    
    known_args, pipeline_args = parser.parse_known_args(argv)

    pipeline_options = PipelineOptions(pipeline_args)
    pipeline_options.view_as(StandardOptions).streaming = True
    pipeline_options.view_as(SetupOptions).save_main_session = True

    with beam.Pipeline(options=pipeline_options) as p:
        events = (
            p 
            | 'Read from PubSub' >> beam.io.ReadFromPubSub(subscription=known_args.input_subscription)
            | 'Parse Event JSON' >> beam.ParDo(ParseEvent())
        )

        aggregated = (
            events
            # 5-minute sliding window, updating every 1 minute
            | 'Window' >> beam.WindowInto(window.SlidingWindows(size=5*60, period=1*60))
            | 'Count per Event' >> beam.CombinePerKey(sum)
        )

        (
            aggregated
            | 'Format for Feature Store' >> beam.ParDo(FormatForBigQuery())
            | 'Write to BigQuery' >> beam.io.WriteToBigQuery(
                table=known_args.output_table,
                schema='event_id:STRING, click_count_5m:INTEGER, feature_timestamp:TIMESTAMP',
                create_disposition=beam.io.BigQueryDisposition.CREATE_IF_NEEDED,
                write_disposition=beam.io.BigQueryDisposition.WRITE_APPEND
            )
        )

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    run()
