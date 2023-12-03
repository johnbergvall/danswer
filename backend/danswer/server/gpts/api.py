from fastapi import APIRouter
from fastapi import Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from danswer.db.chat import create_chat_session
from danswer.db.engine import get_session
from danswer.db.feedback import create_query_event
from danswer.db.feedback import update_query_event_retrieved_documents
from danswer.document_index.factory import get_default_document_index
from danswer.search.models import BaseFilters
from danswer.search.models import SearchType
from danswer.search.request_preprocessing import retrieval_preprocessing
from danswer.search.search_runner import full_chunk_search
from danswer.server.chat.models import NewMessageRequest
from danswer.server.danswer_api.ingestion import api_key_dep
from danswer.utils.logger import setup_logger


logger = setup_logger()


router = APIRouter(prefix="/gpts")


class GptSearchRequest(BaseModel):
    query: str


class GptDocChunk(BaseModel):
    title: str
    content: str
    source_type: str
    link: str


class GptSearchResponse(BaseModel):
    matching_document_chunks: list[GptDocChunk]


@router.post("/document-search")
def handle_search_request(
    search_request: GptSearchRequest,
    _: str | None = Depends(api_key_dep),
    db_session: Session = Depends(get_session),
) -> GptSearchResponse:
    query = search_request.query
    # create record for this query in Postgres
    chat_session = create_chat_session(
        db_session=db_session, description="", user_id=None
    )
    query_event_id = create_query_event(
        query=query,
        chat_session_id=chat_session.id,
        search_type=SearchType.HYBRID,
        llm_answer=None,
        user_id=None,
        db_session=db_session,
    )

    retrieval_request, _, _ = retrieval_preprocessing(
        new_message_request=NewMessageRequest(
            chat_session_id=chat_session.id,
            query=query,
            filters=BaseFilters(),
        ),
        user=None,
        db_session=db_session,
        include_query_intent=False,
    )

    top_chunks, _ = full_chunk_search(
        query=retrieval_request,
        document_index=get_default_document_index(),
    )

    # attach retrieved doc to query row
    update_query_event_retrieved_documents(
        db_session=db_session,
        retrieved_document_ids=[doc.document_id for doc in top_chunks]
        if top_chunks
        else [],
        query_id=query_event_id,
        user_id=None,
    )

    return GptSearchResponse(
        matching_document_chunks=[
            GptDocChunk(
                title=chunk.semantic_identifier,
                content=chunk.content,
                source_type=chunk.source_type,
                link=chunk.source_links.get(0, "") if chunk.source_links else "",
            )
            for chunk in top_chunks
        ],
    )
