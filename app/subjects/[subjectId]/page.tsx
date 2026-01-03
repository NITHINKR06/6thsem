import Link from "next/link"
import { notFound } from "next/navigation"
import { BookOpen, ChevronRight } from "lucide-react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { subjectsData } from "@/lib/subjects-data"

export default async function SubjectDetailPage({
  params,
}: {
  params: Promise<{ subjectId: string }>
}) {
  const { subjectId } = await params
  const subject = subjectsData[subjectId]

  if (!subject) {
    notFound()
  }


  return (
    <div className="min-h-screen bg-background">
      <div className="container mx-auto px-4 py-12">
        {/* Breadcrumb */}
        <nav className="flex items-center space-x-2 text-sm text-muted-foreground mb-8">
          <Link href="/" className="hover:text-foreground">
            Home
          </Link>
          <ChevronRight className="h-4 w-4" />
          <Link href="/subjects" className="hover:text-foreground">
            Subjects
          </Link>
          <ChevronRight className="h-4 w-4" />
          <span className="text-foreground">{subject.title}</span>
        </nav>

        {/* Subject Header */}
        <div className="mb-12">
          <h1 className="text-4xl font-bold mb-4 text-balance">{subject.title}</h1>
          <p className="text-lg text-muted-foreground leading-relaxed max-w-3xl">{subject.description}</p>
        </div>

        {/* Units Grid */}
        <div className="space-y-6">
          <h2 className="text-2xl font-bold">Course Units</h2>
          {subject.units.map((unit, index) => (
            <Card key={unit.id} className="transition-all hover:shadow-lg hover:border-primary/50">
              <CardHeader>
                <div className="flex items-start justify-between gap-4">
                  <div className="flex-1">
                    <div className="flex items-center gap-3 mb-2">
                      <Badge className="text-base px-3 py-1">Unit {index + 1}</Badge>
                      <CardTitle className="text-2xl">{unit.title}</CardTitle>
                    </div>
                    <CardDescription className="text-base leading-relaxed mb-4">{unit.description}</CardDescription>
                  </div>
                  <BookOpen className="h-6 w-6 text-muted-foreground shrink-0" />
                </div>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  <h3 className="font-semibold text-sm text-muted-foreground uppercase tracking-wide">
                    Topics Covered ({unit.topics.length})
                  </h3>
                  <div className="grid sm:grid-cols-2 gap-3">
                    {unit.topics.map((topic) => (
                      <Link
                        key={topic.id}
                        href={`/subjects/${subjectId}/units/${unit.id}/topics/${topic.id}`}
                        className="flex items-center gap-2 p-3 rounded-lg border border-border hover:border-primary hover:bg-accent transition-colors group"
                      >
                        <ChevronRight className="h-4 w-4 text-muted-foreground group-hover:text-primary" />
                        <span className="text-sm font-medium">{topic.title}</span>
                      </Link>
                    ))}
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    </div>
  )
}
