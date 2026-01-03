import { MetadataRoute } from 'next'
import { subjectsData } from '@/lib/subjects-data'

export default function sitemap(): MetadataRoute.Sitemap {
    const baseUrl = 'https://cb6thsem.vercel.app'

    // Static pages
    const staticPages: MetadataRoute.Sitemap = [
        {
            url: baseUrl,
            lastModified: new Date(),
            changeFrequency: 'weekly',
            priority: 1,
        },
        {
            url: `${baseUrl}/subjects`,
            lastModified: new Date(),
            changeFrequency: 'weekly',
            priority: 0.9,
        },
    ]

    // Dynamic subject pages
    const subjectPages: MetadataRoute.Sitemap = Object.values(subjectsData).map((subject) => ({
        url: `${baseUrl}/subjects/${subject.id}`,
        lastModified: new Date(),
        changeFrequency: 'weekly' as const,
        priority: 0.8,
    }))

    // Dynamic unit pages
    const unitPages: MetadataRoute.Sitemap = Object.values(subjectsData).flatMap((subject) =>
        subject.units.map((unit) => ({
            url: `${baseUrl}/subjects/${subject.id}/units/${unit.id}`,
            lastModified: new Date(),
            changeFrequency: 'weekly' as const,
            priority: 0.7,
        }))
    )

    return [...staticPages, ...subjectPages, ...unitPages]
}
