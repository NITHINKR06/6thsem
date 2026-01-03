import TopicContent from "./topic-content"

export default async function TopicPage({
  params,
}: {
  params: Promise<{ subjectId: string; unitId: string; topicId: string }>
}) {
  const { subjectId, unitId, topicId } = await params
  return <TopicContent subjectId={subjectId} unitId={unitId} topicId={topicId} />
}
